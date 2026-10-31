"""SatuSehat (Kemenkes) FHIR R4 client: Fernet-encrypted secrets, OAuth2 token cache, and minimal resources.

Credentials (client_id, client_secret, organization_id, environment) come from the database via the dashboard.
Only the base URLs and SATUSEHAT_ENCRYPTION_KEY come from .env. The client secret is never logged or returned.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests
from cryptography.fernet import Fernet, InvalidToken

ENVIRONMENTS: dict[str, tuple[str, str]] = {
    "dev": ("SATUSEHAT_BASE_URL_DEV", "https://api-satusehat-dev.dto.kemkes.go.id"),
    "staging": ("SATUSEHAT_BASE_URL_STAGING", "https://api-satusehat-stg.dto.kemkes.go.id"),
    "production": ("SATUSEHAT_BASE_URL_PRODUCTION", "https://api-satusehat.kemkes.go.id"),
}
NIK_SYSTEM = "https://fhir.kemkes.go.id/id/nik"
LOCATION_SYSTEM_TEMPLATE = "http://sys-ids.kemkes.go.id/location/{organization_id}"
REQUEST_TIMEOUT = 20
AUTH_FAILURE_COOLDOWN = 30  # seconds; avoids hammering the token endpoint after a rejection


class SatuSehatError(Exception):
    """Error with a user-safe message and the HTTP status the API should return."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass(frozen=True)
class Config:
    env: str
    client_id: str
    client_secret: str
    organization_id: str


# --------------------------------------------------------------------------- encryption


def _fernet() -> Fernet:
    key = os.getenv("SATUSEHAT_ENCRYPTION_KEY", "").strip()
    if not key:
        raise SatuSehatError("SATUSEHAT_ENCRYPTION_KEY belum diisi di .env (lihat .env.example).", 503)
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError):
        raise SatuSehatError("SATUSEHAT_ENCRYPTION_KEY tidak valid (harus kunci Fernet base64 32 byte).", 503) from None


def encryption_ready() -> bool:
    try:
        _fernet()
        return True
    except SatuSehatError:
        return False


def encrypt(plain: str) -> str:
    return _fernet().encrypt(plain.encode()).decode()


def decrypt(token: str) -> str:
    try:
        return _fernet().decrypt(token.encode()).decode()
    except InvalidToken:
        raise SatuSehatError(
            "Data terenkripsi tidak bisa dibuka (SATUSEHAT_ENCRYPTION_KEY berubah). Isi ulang nilainya.", 409
        ) from None


# --------------------------------------------------------------------------- HTTP / OAuth2


def base_url(env: str) -> str:
    if env not in ENVIRONMENTS:
        raise SatuSehatError(f"Lingkungan SatuSehat tidak dikenal: {env}", 400)
    var, default = ENVIRONMENTS[env]
    return (os.getenv(var, "").strip() or default).rstrip("/")


_token_lock = threading.Lock()
_tokens: dict[tuple[str, str], tuple[str, float]] = {}
_cooldown: dict[tuple[str, str], float] = {}


def clear_token_cache() -> None:
    with _token_lock:
        _tokens.clear()
        _cooldown.clear()


def get_token(cfg: Config, force: bool = False) -> str:
    key = (cfg.env, cfg.client_id)
    with _token_lock:
        cached = _tokens.get(key)
        if cached and not force and cached[1] - 60 > time.time():
            return cached[0]
        if _cooldown.get(key, 0) > time.time():
            raise SatuSehatError("Autentikasi SatuSehat baru saja ditolak; coba lagi dalam beberapa detik.", 429)
        url = f"{base_url(cfg.env)}/oauth2/v1/accesstoken"
        try:
            response = requests.post(
                url,
                params={"grant_type": "client_credentials"},
                data={"client_id": cfg.client_id, "client_secret": cfg.client_secret},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise SatuSehatError(f"Tidak dapat terhubung ke SatuSehat ({type(exc).__name__}).", 502) from None
        if response.status_code != 200:
            _cooldown[key] = time.time() + AUTH_FAILURE_COOLDOWN
            raise SatuSehatError(
                f"Autentikasi ditolak SatuSehat (HTTP {response.status_code}). "
                "Periksa Client ID, Client Secret, dan lingkungan.",
                401,
            )
        try:
            payload = response.json()
            token = str(payload["access_token"])
            expires_in = int(payload.get("expires_in", 3600))
        except (ValueError, KeyError, TypeError):
            raise SatuSehatError("Respons token SatuSehat tidak dikenali.", 502) from None
        _tokens[key] = (token, time.time() + expires_in)
        _cooldown.pop(key, None)
        return token


def _outcome_message(response: requests.Response) -> str:
    try:
        issues = response.json().get("issue") or []
        if issues:
            first = issues[0]
            text = (first.get("details") or {}).get("text") or first.get("diagnostics") or first.get("code")
            if text:
                return str(text)[:300]
    except ValueError:
        pass
    return f"HTTP {response.status_code}"


def fhir_request(cfg: Config, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    url = f"{base_url(cfg.env)}/fhir-r4/v1/{path.lstrip('/')}"
    for attempt in (0, 1):
        headers = {"Authorization": f"Bearer {get_token(cfg, force=attempt == 1)}", "Accept": "application/fhir+json"}
        if "json" in kwargs:
            headers["Content-Type"] = "application/fhir+json"
        try:
            response = requests.request(method, url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.RequestException as exc:
            raise SatuSehatError(f"Tidak dapat terhubung ke SatuSehat ({type(exc).__name__}).", 502) from None
        if response.status_code == 401 and attempt == 0:
            continue  # token may have been revoked early; refresh once
        break
    if response.status_code == 404:
        raise SatuSehatError("Data tidak ditemukan di SatuSehat.", 404)
    if not response.ok:
        raise SatuSehatError(f"SatuSehat menolak permintaan: {_outcome_message(response)}", 502)
    try:
        return response.json()
    except ValueError:
        raise SatuSehatError("Respons FHIR SatuSehat bukan JSON.", 502) from None


# --------------------------------------------------------------------------- resources


def test_connection(cfg: Config) -> str:
    """Fetch the token, then read Organization/{id}. Returns the organization display name."""
    organization = fhir_request(cfg, "GET", f"Organization/{cfg.organization_id}")
    return str(organization.get("name") or cfg.organization_id)


def find_practitioner_id(cfg: Config, nik: str) -> str | None:
    bundle = fhir_request(cfg, "GET", "Practitioner", params={"identifier": f"{NIK_SYSTEM}|{nik}"})
    for entry in bundle.get("entry") or []:
        resource = entry.get("resource") or {}
        if resource.get("resourceType") == "Practitioner" and resource.get("id"):
            return str(resource["id"])
    return None


def ensure_location(cfg: Config, code: str, name: str) -> str:
    """Return the Location id for (code), creating it under the configured Organization if missing."""
    system = LOCATION_SYSTEM_TEMPLATE.format(organization_id=cfg.organization_id)
    found = fhir_request(
        cfg,
        "GET",
        "Location",
        params={"identifier": f"{system}|{code}", "organization": cfg.organization_id},
    )
    for entry in found.get("entry") or []:
        resource = entry.get("resource") or {}
        if resource.get("resourceType") == "Location" and resource.get("id"):
            return str(resource["id"])
    created = fhir_request(
        cfg,
        "POST",
        "Location",
        json={
            "resourceType": "Location",
            "identifier": [{"system": system, "value": code}],
            "status": "active",
            "name": name,
            "description": f"{name} (kamera StaffLens AI)",
            "mode": "instance",
            "physicalType": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/location-physical-type",
                        "code": "ro",
                        "display": "Room",
                    }
                ]
            },
            "managingOrganization": {"reference": f"Organization/{cfg.organization_id}"},
        },
    )
    if not created.get("id"):
        raise SatuSehatError("SatuSehat tidak mengembalikan ID Location.", 502)
    return str(created["id"])
