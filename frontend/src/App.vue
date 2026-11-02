<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import './style.css'
import './employee-modal.css'
import './customer-photos.css'

const authenticated = ref(false)
const checking = ref(true)
const password = ref('')
const loginError = ref('')
const busy = ref(false)
const page = ref('Ringkasan')
const activityTab = ref('visitors')
const eventFilter = ref('all')
const settings = ref({
  video_source: 0,
  performance_level: 5,
  camera_width: null,
  camera_height: null,
  customer_min_dwell_seconds: 0.5,
})
const resources = ref(null)
const settingsNotice = ref('')
const satusehat = ref(null)
const satusehatForm = ref({ env: 'dev', client_id: '', organization_id: '', client_secret: '', location_name: '' })
const satusehatNotice = ref('')
const satusehatError = ref('')
const satusehatBusy = ref('')
const visionAiBusy = ref(false)
const overview = ref({
  customers_today: 0,
  returning_customers_today: 0,
  total_customer_visits_today: 0,
  staff_present_today: 0,
  employees: 0,
  embeddings: 0,
  events_today: 0,
  events: 0,
  camera_id: 'local-webcam-01',
  vision_ai: { enabled: false, model: null, configured: false, status: 'checking', message: '' },
})
const visitors = ref([])
const events = ref([])
const employees = ref([])
const camera = ref({
  running: false,
  fps: 0,
  has_frame: false,
  error: null,
  live_staff: 0,
  live_customers: 0,
  live_observing: 0,
})
const dataError = ref('')
const lastUpdated = ref(null)
const hasRenderedFrame = ref(false)
const cameraCanvasRef = ref(null)
let cameraWs = null
let wsReconnectTimer = null
let wsSessionToken = 0
let wsConnected = false
let renderEpoch = 0
let frameRenderBusy = false
let pendingFrameBlob = null
const search = ref('')
let timer
let resourcesTimer

function clearCanvas() {
  renderEpoch++
  hasRenderedFrame.value = false
  pendingFrameBlob = null
  const canvas = cameraCanvasRef.value
  if (canvas) {
    const ctx = canvas.getContext('2d', { alpha: false })
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
    canvas.width = 0
    canvas.height = 0
  }
}

function applyCameraStatus(nextStatus) {
  if (!nextStatus) return
  const prevGen = camera.value.generation || 0
  const nextGen = nextStatus.generation || 0
  const genChanged = Boolean(prevGen && nextGen && prevGen !== nextGen)
  camera.value = {
    running: Boolean(nextStatus.running),
    stopping: Boolean(nextStatus.stopping),
    fps: nextStatus.fps ?? 0,
    has_frame: Boolean(nextStatus.has_frame),
    generation: nextGen,
    frame_version: nextStatus.frame_version ?? 0,
    error: nextStatus.error ?? null,
    active_config: nextStatus.active_config ?? null,
    live_staff: nextStatus.live_staff ?? 0,
    live_customers: nextStatus.live_customers ?? 0,
    live_observing: nextStatus.live_observing ?? 0,
  }
  if (!camera.value.running || !camera.value.has_frame || genChanged) {
    clearCanvas()
  }
}

async function renderFrameToCanvas(blob) {
  if (!blob || page.value !== 'Ringkasan' || !camera.value.running || !camera.value.has_frame) return
  pendingFrameBlob = { blob, epoch: renderEpoch }
  if (frameRenderBusy) return
  frameRenderBusy = true
  try {
    while (pendingFrameBlob && page.value === 'Ringkasan' && camera.value.running && camera.value.has_frame) {
      const { blob: currentBlob, epoch: myEpoch } = pendingFrameBlob
      pendingFrameBlob = null
      if (myEpoch !== renderEpoch) continue
      if (!cameraCanvasRef.value && page.value === 'Ringkasan') {
        await nextTick()
      }
      const canvas = cameraCanvasRef.value
      if (myEpoch !== renderEpoch) continue
      if (!canvas || page.value !== 'Ringkasan' || !camera.value.running || !camera.value.has_frame) break
      let bitmap = null
      try {
        bitmap = await createImageBitmap(currentBlob)
        if (
          myEpoch === renderEpoch &&
          cameraCanvasRef.value === canvas &&
          camera.value.running &&
          camera.value.has_frame &&
          page.value === 'Ringkasan'
        ) {
          if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
            canvas.width = bitmap.width
            canvas.height = bitmap.height
          }
          const ctx = canvas.getContext('2d', { alpha: false })
          if (ctx) {
            ctx.drawImage(bitmap, 0, 0)
            hasRenderedFrame.value = true
          }
        }
      } catch (_) {
        /* skip corrupted frame */
      } finally {
        if (bitmap) bitmap.close()
      }
    }
  } finally {
    if (
      !pendingFrameBlob ||
      pendingFrameBlob.epoch !== renderEpoch ||
      page.value !== 'Ringkasan' ||
      !camera.value.running ||
      !camera.value.has_frame
    ) {
      pendingFrameBlob = null
    }
    frameRenderBusy = false
  }
}

function syncCameraSubscription(forceStatus = false, forceFrame = false) {
  if (cameraWs && cameraWs.readyState === WebSocket.OPEN) {
    try {
      cameraWs.send(
        JSON.stringify({
          subscribe_frames: page.value === 'Ringkasan',
          force_frame: Boolean(forceFrame),
          force_status: Boolean(forceStatus),
        }),
      )
    } catch (_) {
      /* ignore send error */
    }
  }
}

function disconnectCameraSocket() {
  wsSessionToken++
  wsConnected = false
  if (wsReconnectTimer) {
    clearTimeout(wsReconnectTimer)
    wsReconnectTimer = null
  }
  if (cameraWs) {
    const ws = cameraWs
    cameraWs = null
    ws.onopen = null
    ws.onmessage = null
    ws.onerror = null
    ws.onclose = null
    try {
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close(1000, 'client disconnect')
      }
    } catch (_) {
      /* ignore close error */
    }
  }
  clearCanvas()
}

function connectCameraSocket(forceStatus = false) {
  if (!authenticated.value) return
  if (cameraWs && (cameraWs.readyState === WebSocket.OPEN || cameraWs.readyState === WebSocket.CONNECTING)) {
    if (forceStatus) syncCameraSubscription(true, page.value === 'Ringkasan')
    return
  }
  if (wsReconnectTimer) {
    clearTimeout(wsReconnectTimer)
    wsReconnectTimer = null
  }
  const myToken = ++wsSessionToken
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const subFlag = page.value === 'Ringkasan' ? 1 : 0
  const ws = new WebSocket(`${proto}//${window.location.host}/api/camera/ws?subscribe=${subFlag}`)
  ws.binaryType = 'blob'
  cameraWs = ws

  ws.onopen = () => {
    if (myToken !== wsSessionToken) return
    wsConnected = true
    syncCameraSubscription(true, page.value === 'Ringkasan')
  }

  ws.onmessage = event => {
    if (myToken !== wsSessionToken) return
    if (typeof event.data === 'string') {
      try {
        const payload = JSON.parse(event.data)
        if (payload.type === 'status') {
          applyCameraStatus(payload)
        }
      } catch (_) {
        /* ignore malformed json */
      }
    } else if (event.data instanceof Blob) {
      if (page.value === 'Ringkasan') {
        renderFrameToCanvas(event.data)
      }
    }
  }

  ws.onerror = () => {
    /* handled by onclose */
  }

  ws.onclose = async event => {
    if (myToken !== wsSessionToken) return
    wsConnected = false
    cameraWs = null
    clearCanvas()
    if (event.code === 4401 || event.code === 1008) {
      authenticated.value = false
      stopPolling()
      return
    }
    if (event.code === 1006 && authenticated.value) {
      try {
        const sess = await api('/auth/session')
        if (!sess.authenticated) {
          authenticated.value = false
          stopPolling()
          return
        }
      } catch (_) {
        /* server unreachable; retry on timer */
      }
    }
    if (authenticated.value) {
      wsReconnectTimer = setTimeout(() => {
        if (myToken === wsSessionToken && authenticated.value) connectCameraSocket(true)
      }, 1000)
    }
  }
}

function selectPage(targetPage) {
  if (page.value === targetPage) return
  if (page.value === 'Ringkasan' && targetPage !== 'Ringkasan') {
    clearCanvas()
  }
  page.value = targetPage
  search.value = ''
  const enteringSummary = targetPage === 'Ringkasan'
  syncCameraSubscription(enteringSummary, enteringSummary)
}

// Employee Modal & Photo Upload State
const showEmployeeModal = ref(false)
const modalLoading = ref(false)
const modalSaving = ref(false)
const modalError = ref('')
const modalNotice = ref('')
const modalEmployee = ref({
  id: null,
  code: '',
  name: '',
  role: 'Karyawan',
  active: true,
  folder: '',
  embeddings: 0,
  photos: [],
})
const photosToDelete = ref([])
const pendingUploads = ref([])
const cameraStream = ref(null)
const cameraVideoRef = ref(null)
const cameraCaptureOpen = ref(false)
const employeeDeleting = ref(false)
const customerPhotoModal = ref({ open: false, customer: null, photos: [], loading: false, error: '' })
const showReviewed = ref(false)
const reviewedVisitors = ref([])
const customerReview = ref({ open: false, customer: null, reason: '', note: '', busy: false, error: '' })
const reviewReasons = [
  ['not_person', 'Bukan orang'], ['bad_crop', 'Crop salah'], ['object', 'Tangan / jam / objek'],
  ['blur', 'Foto blur'], ['duplicate', 'Duplikat customer'], ['employee', 'Salah klasifikasi karyawan'], ['other', 'Lainnya'],
]

const navItems = ['Ringkasan', 'Customer', 'Pegawai', 'Pengaturan']

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, { credentials: 'same-origin', cache: 'no-store', ...options })
  if (response.status === 401 && path !== '/auth/login') {
    authenticated.value = false
    disconnectCameraSocket()
    stopPolling()
    throw new Error('Sesi berakhir. Masuk kembali.')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Permintaan gagal (${response.status})`)
  }
  return response.json()
}

async function refresh(forceStream = false) {
  if (!authenticated.value) return
  try {
    const bundle = await api('/dashboard')
    overview.value = bundle.overview
    visitors.value = bundle.visitors || []
    events.value = bundle.events || []
    employees.value = bundle.employees || []
    if (!wsConnected && bundle.camera) {
      applyCameraStatus(bundle.camera)
    } else if (bundle.camera && camera.value.running && bundle.camera.running) {
      camera.value.fps = bundle.camera.fps ?? camera.value.fps
    }
    connectCameraSocket(forceStream === true)
    lastUpdated.value = new Date()
    dataError.value = ''
  } catch (error) {
    if (!wsConnected || !authenticated.value) clearCanvas()
    dataError.value = error.message
  }
}

async function refreshResources() {
  if (!authenticated.value) return
  try {
    resources.value = await api('/system/resources')
  } catch (_) {
    resources.value = null
  }
}

async function loadSettings() {
  if (!authenticated.value) return
  try {
    settings.value = await api('/camera/settings')
  } catch (error) {
    dataError.value = error.message
  }
}

async function saveSettings() {
  busy.value = true
  settingsNotice.value = ''
  try {
    const result = await api('/camera/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        video_source: Number(settings.value.video_source),
        performance_level: Number(settings.value.performance_level),
        camera_width:
          settings.value.camera_width === '' || settings.value.camera_width == null
            ? null
            : Number(settings.value.camera_width),
        camera_height:
          settings.value.camera_height === '' || settings.value.camera_height == null
            ? null
            : Number(settings.value.camera_height),
        customer_min_dwell_seconds: Number(settings.value.customer_min_dwell_seconds),
      }),
    })
    settings.value = result.settings
    settingsNotice.value = result.running
      ? 'Tersimpan. Worker kamera dimulai ulang; server tetap berjalan.'
      : 'Tersimpan. Berlaku saat kamera dinyalakan.'
    await refresh()
  } catch (error) {
    dataError.value = error.message
  } finally {
    busy.value = false
  }
}

function applySatuSehat(view) {
  satusehat.value = view
  satusehatForm.value = {
    env: view.env,
    client_id: view.client_id,
    organization_id: view.organization_id,
    client_secret: '',
    location_name: view.location_name,
  }
}

async function loadSatuSehat() {
  if (!authenticated.value) return
  try {
    applySatuSehat(await api('/satusehat/settings'))
  } catch (error) {
    satusehatError.value = error.message
  }
}

async function runSatuSehat(kind, action) {
  satusehatBusy.value = kind
  satusehatError.value = ''
  satusehatNotice.value = ''
  try {
    await action()
  } catch (error) {
    satusehatError.value = error.message
  } finally {
    satusehatBusy.value = ''
  }
}

function saveSatuSehat() {
  return runSatuSehat('save', async () => {
    applySatuSehat(
      await api('/satusehat/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(satusehatForm.value),
      }),
    )
    satusehatNotice.value = 'Tersimpan. Client Secret terenkripsi di database.'
  })
}

async function testVisionAi() {
  visionAiBusy.value = true
  try {
    await api('/vision-ai/test', { method: 'POST' })
    await refresh()
  } catch (error) {
    dataError.value = error.message
  } finally {
    visionAiBusy.value = false
  }
}

function testSatuSehat() {
  return runSatuSehat('test', async () => {
    const result = await api('/satusehat/test', { method: 'POST' })
    satusehatNotice.value = result.message
    await loadSatuSehat()
  })
}

function syncSatuSehatLocation() {
  return runSatuSehat('location', async () => {
    // Persist any edited location name first so the sync uses what is on screen.
    applySatuSehat(
      await api('/satusehat/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(satusehatForm.value),
      }),
    )
    const result = await api('/satusehat/location/sync', { method: 'POST' })
    satusehatNotice.value = `Location tersinkron (ID ${result.location_id}).`
    await loadSatuSehat()
  })
}

async function syncEmployeeSatuSehat() {
  modalSaving.value = true
  modalError.value = ''
  modalNotice.value = ''
  try {
    const result = await api(`/employees/${modalEmployee.value.id}/satusehat/sync`, { method: 'POST' })
    modalEmployee.value = { ...modalEmployee.value, ...result }
    modalNotice.value = `Practitioner SatuSehat ditemukan (ID ${result.satusehat_practitioner_id}).`
    await refresh()
  } catch (error) {
    modalError.value = error.message
  } finally {
    modalSaving.value = false
  }
}

function suggestNextEmployeeCode() {
  const nums = employees.value
    .map(e => {
      const m = String(e.code || '').match(/(\d+)$/)
      return m ? Number(m[1]) : 0
    })
    .filter(n => n > 0)
  const next = (nums.length ? Math.max(...nums) : employees.value.length) + 1
  return `EMP-${String(next).padStart(3, '0')}`
}

const folderSlugPreview = computed(() => {
  const raw = (modalEmployee.value.name || 'pegawai_baru').trim().toLowerCase()
  const slug = raw.replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') || 'pegawai_baru'
  return `db_karyawan/${slug}`
})

async function openEmployeeModal(emp = null) {
  modalError.value = ''
  modalNotice.value = ''
  photosToDelete.value = []
  pendingUploads.value = []
  showEmployeeModal.value = true

  if (!emp) {
    modalEmployee.value = {
      id: null,
      code: suggestNextEmployeeCode(),
      name: '',
      role: 'Karyawan',
      active: true,
      folder: '',
      embeddings: 0,
      photos: [],
      nik: '',
      nik_clear: false,
    }
    return
  }

  modalLoading.value = true
  modalEmployee.value = {
    id: emp.id,
    code: emp.code,
    name: emp.name,
    role: emp.role,
    active: emp.active,
    folder: emp.folder || '',
    embeddings: emp.embeddings || 0,
    photos: [],
    nik: '',
    nik_clear: false,
    has_nik: emp.has_nik,
    nik_masked: emp.nik_masked,
    satusehat_practitioner_id: emp.satusehat_practitioner_id,
  }
  try {
    const detail = await api(`/employees/${emp.id}`)
    modalEmployee.value = { ...detail, nik: '', nik_clear: false }
  } catch (error) {
    modalError.value = error.message
  } finally {
    modalLoading.value = false
  }
}

function closeEmployeeModal() {
  if (modalSaving.value) return
  stopPhotoCamera()
  showEmployeeModal.value = false
  modalError.value = ''
  modalNotice.value = ''
  pendingUploads.value = []
  photosToDelete.value = []
}

function toggleDeleteExistingPhoto(photoName) {
  const idx = photosToDelete.value.indexOf(photoName)
  if (idx >= 0) photosToDelete.value.splice(idx, 1)
  else photosToDelete.value.push(photoName)
}

function removePendingUpload(index) {
  pendingUploads.value.splice(index, 1)
}

async function startPhotoCamera() {
  modalError.value = ''
  if (!navigator.mediaDevices?.getUserMedia) {
    modalError.value = 'Kamera browser tidak tersedia. Gunakan HTTPS atau localhost, atau pilih foto dari perangkat.'
    return
  }
  try {
    cameraStream.value = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
    cameraCaptureOpen.value = true
    await nextTick()
    cameraVideoRef.value.srcObject = cameraStream.value
    await cameraVideoRef.value.play()
  } catch (error) {
    stopPhotoCamera()
    modalError.value = `Tidak dapat membuka kamera: ${error.message || 'izin kamera ditolak'}`
  }
}

function stopPhotoCamera() {
  cameraStream.value?.getTracks().forEach(track => track.stop())
  cameraStream.value = null
  cameraCaptureOpen.value = false
}

function capturePhoto() {
  const video = cameraVideoRef.value
  if (!video || !video.videoWidth) return
  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth
  canvas.height = video.videoHeight
  canvas.getContext('2d').drawImage(video, 0, 0)
  const dataUrl = canvas.toDataURL('image/jpeg', 0.92)
  pendingUploads.value.push({
    filename: `capture-${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`,
    size_bytes: Math.round((dataUrl.length * 3) / 4),
    data_url: dataUrl,
  })
  modalError.value = ''
}

function onSelectPhotoFiles(event) {
  const files = Array.from(event.target.files || [])
  modalError.value = ''
  for (const file of files) {
    if (!/\.(jpe?g|png)$/i.test(file.name)) {
      modalError.value = `File ${file.name} dilewati (gunakan format .jpg, .jpeg, atau .png).`
      continue
    }
    const reader = new FileReader()
    reader.onload = () => {
      if (typeof reader.result === 'string') {
        pendingUploads.value.push({
          filename: file.name,
          size_bytes: file.size,
          data_url: reader.result,
        })
      }
    }
    reader.readAsDataURL(file)
  }
  event.target.value = ''
}

async function saveEmployeeModal(reprocessOnly = false) {
  modalSaving.value = true
  modalError.value = ''
  modalNotice.value = ''
  try {
    const isNew = modalEmployee.value.id == null
    const path = isNew ? '/employees' : `/employees/${modalEmployee.value.id}`
    const method = isNew ? 'POST' : 'PUT'
    const result = await api(path, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code: modalEmployee.value.code,
        name: modalEmployee.value.name,
        role: modalEmployee.value.role,
        active: Boolean(modalEmployee.value.active),
        nik: modalEmployee.value.nik_clear ? '' : modalEmployee.value.nik?.trim() || undefined,
        new_photos: pendingUploads.value.map(item => ({
          filename: item.filename,
          data_url: item.data_url,
        })),
        delete_photos: photosToDelete.value,
        reprocess_embeddings: Boolean(reprocessOnly),
      }),
    })
    modalEmployee.value = { ...result, nik: '', nik_clear: false }
    pendingUploads.value = []
    photosToDelete.value = []
    const skippedCount = (result.skipped_photos || []).length
    const skipNote = skippedCount
      ? ` (${skippedCount} foto dilewati karena wajah tidak tunggal/jelas)`
      : ''
    modalNotice.value = `Tersimpan di ${result.folder}. ${result.embeddings} vektor wajah aktif di PostgreSQL${skipNote}.`
    await refresh()
  } catch (error) {
    modalError.value = error.message
  } finally {
    modalSaving.value = false
  }
}

async function deleteEmployee() {
  const employee = modalEmployee.value
  if (!employee.id || employeeDeleting.value || modalSaving.value) return
  const confirmed = window.confirm(
    `Hapus pegawai ${employee.name} (${employee.code})? Foto dan data wajahnya akan dihapus permanen. Riwayat kamera tetap disimpan tanpa nama pegawai.`,
  )
  if (!confirmed) return
  employeeDeleting.value = true
  modalError.value = ''
  try {
    await api(`/employees/${employee.id}`, { method: 'DELETE' })
    showEmployeeModal.value = false
    await refresh()
  } catch (error) {
    modalError.value = error.message
  } finally {
    employeeDeleting.value = false
  }
}

function stopPolling() {
  clearInterval(timer)
  clearInterval(resourcesTimer)
  timer = undefined
  resourcesTimer = undefined
}

function startPolling() {
  stopPolling()
  connectCameraSocket(true)
  refresh()
  refreshResources()
  loadSettings()
  loadSatuSehat()
  timer = setInterval(refresh, 3000)
  resourcesTimer = setInterval(refreshResources, 5000)
}

async function login() {
  busy.value = true
  loginError.value = ''
  try {
    await api('/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ password: password.value }),
    })
    password.value = ''
    authenticated.value = true
    startPolling()
  } catch (error) {
    loginError.value = error.message
  } finally {
    busy.value = false
  }
}

async function logout() {
  disconnectCameraSocket()
  authenticated.value = false
  stopPolling()
  try {
    await api('/auth/logout', { method: 'POST' })
  } catch (_) {
    /* clear local view regardless */
  }
}

async function toggleCamera() {
  if (busy.value || camera.value.stopping) return
  busy.value = true
  try {
    const wasRunning = camera.value.running
    if (wasRunning) clearCanvas()
    await api(wasRunning ? '/camera/stop' : '/camera/start', { method: 'POST' })
    await refresh(true)
  } catch (error) {
    dataError.value = error.message
  } finally {
    busy.value = false
  }
}

async function openCustomerPhotos(customer) {
  customerPhotoModal.value = { open: true, customer, photos: [], loading: true, error: '' }
  try {
    const result = await api(`/visitors/${customer.id}/photos`)
    customerPhotoModal.value.photos = result.photos || []
  } catch (error) {
    customerPhotoModal.value.error = error.message
  } finally {
    customerPhotoModal.value.loading = false
  }
}

function closeCustomerPhotos() {
  customerPhotoModal.value = { open: false, customer: null, photos: [], loading: false, error: '' }
}

function openCustomerReview(customer) {
  customerReview.value = { open: true, customer, reason: '', note: '', busy: false, error: '' }
}

async function submitCustomerReview() {
  const review = customerReview.value
  if (!review.reason || review.busy) return
  review.busy = true
  try {
    await api(`/visitors/${review.customer.id}/review`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ reason_code: review.reason, note: review.note || null }),
    })
    customerReview.value.open = false
    closeCustomerPhotos()
    await refresh()
  } catch (error) {
    review.error = error.message
  } finally {
    review.busy = false
  }
}

async function loadReviewed() {
  if (!showReviewed.value) return
  const result = await api('/visitors?include_reviewed=true')
  reviewedVisitors.value = result.items || []
}

async function restoreReviewed(customer) {
  await api(`/visitors/${customer.id}/restore`, { method: 'POST' })
  await loadReviewed(); await refresh()
}

async function permanentlyDeleteReviewed(customer) {
  if (!window.confirm(`Hapus permanen ${customer.code}?`)) return
  await api(`/visitors/${customer.id}`, { method: 'DELETE' })
  await loadReviewed()
}

const filteredVisitors = computed(() =>
  visitors.value.filter(item =>
    `${item.label} ${item.code} ${item.attributes || ''}`.toLowerCase().includes(search.value.toLowerCase()),
  ),
)

const filteredEvents = computed(() =>
  events.value.filter(item => {
    if (eventFilter.value !== 'all' && item.kind !== eventFilter.value) return false
    return `${item.label} ${item.camera_id} ${item.track_id} ${item.visitor_code || ''}`
      .toLowerCase()
      .includes(search.value.toLowerCase())
  }),
)

const filteredEmployees = computed(() =>
  employees.value.filter(item =>
    `${item.name} ${item.code} ${item.role} ${item.folder || ''}`
      .toLowerCase()
      .includes(search.value.toLowerCase()),
  ),
)

const absentStaffCount = computed(() => Math.max(0, overview.value.employees - overview.value.staff_present_today))

function pageSubtitle(current) {
  if (current === 'Ringkasan') return 'Hitungan customer unik harian dan kehadiran karyawan aktif di lokasi.'
  if (current === 'Customer') return 'Daftar pengunjung unik hari ini (1 hari = 1 customer) dan log kamera.'
  if (current === 'Pegawai')
    return 'Kelola pegawai, akses galeri foto di db_karyawan, dan registrasi vektor wajah otomatis.'
  return 'Pengaturan perangkat kamera, deduplikasi pengunjung, dan pemakaian sumber daya.'
}

function timeLabel(value) {
  return value
    ? new Intl.DateTimeFormat('id-ID', { dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value))
    : '-'
}

function clockLabel(value) {
  return value ? new Intl.DateTimeFormat('id-ID', { timeStyle: 'short' }).format(new Date(value)) : '-'
}

function percent(value) {
  return value == null ? '-' : `${(value * 100).toFixed(1)}%`
}

function gigabytes(bytes) {
  return bytes == null ? '-' : `${(bytes / 1024 ** 3).toFixed(1)} GB`
}

function kilobytes(bytes) {
  return bytes == null ? '-' : `${Math.max(1, Math.round(bytes / 1024))} KB`
}

onMounted(async () => {
  try {
    const response = await api('/auth/session')
    authenticated.value = response.authenticated
    if (authenticated.value) startPolling()
  } catch (error) {
    loginError.value = error.message
  } finally {
    checking.value = false
  }
})

onUnmounted(() => {
  stopPolling()
  disconnectCameraSocket()
})

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    stopPolling()
    disconnectCameraSocket()
  })
}
</script>

<template>
  <div v-if="checking" class="boot">Menyiapkan StaffLens…</div>
  <main v-else-if="!authenticated" class="login-layout">
    <div class="login-intro">
      <div class="wordmark">
        <span class="brand-mark" aria-hidden="true"><span></span></span> StaffLens <span class="brand-ai">AI</span>
      </div>
      <div class="intro-bottom">
        <div class="intro-rule"></div>
        <p>Hitungan customer &amp; staf<br />untuk kamera kerja Anda.</p>
        <span>Satu customer dihitung sekali sehari, terpisah dari karyawan aktif.</span>
      </div>
    </div>
    <div class="login-pane">
      <form class="login-form" @submit.prevent="login">
        <span class="scope-meta">Portal terbatas</span>
        <h1>Masuk ke ruang kendali.</h1>
        <p>Masukkan kata sandi portal untuk memantau jumlah customer harian dan kehadiran pegawai.</p>
        <label for="password">Kata sandi</label>
        <input
          id="password"
          v-model="password"
          type="password"
          autocomplete="current-password"
          placeholder="Kata sandi portal"
          required
          autofocus
        />
        <p v-if="loginError" class="form-error" role="alert">{{ loginError }}</p>
        <button class="button primary login-button" type="submit" :disabled="busy">
          {{ busy ? 'Memeriksa…' : 'Masuk ke dashboard' }} <span aria-hidden="true">→</span>
        </button>
        <div class="login-foot">Akses khusus operator · Vektor wajah &amp; pakaian tetap di server.</div>
      </form>
    </div>
  </main>

  <div v-else class="app-shell">
    <aside class="sidebar">
      <div class="wordmark">
        <span class="brand-mark" aria-hidden="true"><span></span></span> StaffLens <span class="brand-ai">AI</span>
      </div>
      <div class="workspace"><span class="workspace-dot"></span> QDerma · Lokal</div>
      <nav aria-label="Navigasi utama">
        <span class="nav-label">Operasional</span>
        <button
          v-for="item in navItems"
          :key="item"
          class="nav-item"
          :class="{ selected: page === item }"
          @click="selectPage(item)"
        >
          <span>{{ item }}</span>
          <span v-if="item === 'Customer'" class="nav-count">{{ overview.customers_today }}</span>
          <span v-else-if="item === 'Pegawai'" class="nav-count">
            {{ overview.staff_present_today }}/{{ overview.employees }}
          </span>
        </button>
      </nav>
      <div class="sidebar-bottom">
        <span class="sidebar-caption">StaffLens / Hitungan Harian</span>
        <button class="text-button" @click="logout">Keluar <span aria-hidden="true">↗</span></button>
      </div>
    </aside>

    <div class="main-column">
      <header class="topbar">
        <div class="mobile-brand">
          <span class="brand-mark" aria-hidden="true"><span></span></span> StaffLens AI
        </div>
        <span class="topbar-context">Operasional / {{ page }}</span>
        <div class="topbar-right">
          <span class="status-dot" :class="{ on: camera.running }"></span>
          {{ camera.running ? `Kamera aktif · ${camera.fps} FPS` : 'Kamera nonaktif' }}
        </div>
      </header>

      <div class="mobile-nav">
        <button
          v-for="item in navItems"
          :key="item"
          :class="{ selected: page === item }"
          @click="selectPage(item)"
        >
          {{ item }}
        </button>
        <button @click="logout">Keluar</button>
      </div>

      <main class="sheet">
        <div class="page-heading">
          <div>
            <span class="scope-meta">StaffLens / {{ overview.camera_id }}</span>
            <h1>{{ page }}</h1>
            <p>{{ pageSubtitle(page) }}</p>
          </div>
          <div class="heading-actions">
            <button v-if="page === 'Pegawai'" class="button primary" @click="openEmployeeModal(null)">
              + Tambah pegawai
            </button>
            <button class="button secondary" @click="refresh(true)">
              Perbarui data <span aria-hidden="true">↗</span>
            </button>
          </div>
        </div>

        <p v-if="dataError" class="error-notice" role="alert">
          {{ dataError }}
          <button class="text-button" @click="refresh(true)">Coba lagi</button>
        </p>

        <template v-if="page === 'Ringkasan'">
          <section class="metric-strip" aria-label="Metrik utama hari ini">
            <div class="metric-lead">
              <span>Customer unik hari ini</span>
              <strong>{{ overview.customers_today }}</strong>
              <small>
                {{ overview.returning_customers_today }} kunjungan ulang · {{ overview.total_customer_visits_today }} total kedatangan
              </small>
            </div>
            <div>
              <span>Karyawan hadir hari ini</span>
              <strong>
                {{ overview.staff_present_today }}
                <em class="metric-unit">/ {{ overview.employees }}</em>
              </strong>
              <small>{{ absentStaffCount }} belum terdeteksi · {{ overview.embeddings }} ref. wajah</small>
            </div>
            <div>
              <span>Subjek di kamera saat ini</span>
              <strong>
                {{ camera.running ? (camera.live_customers || 0) + (camera.live_staff || 0) : 0 }}
              </strong>
              <small>
                {{
                  camera.running
                    ? `${camera.live_customers || 0} customer · ${camera.live_staff || 0} karyawan aktif`
                    : 'Nyalakan kamera untuk pantauan langsung'
                }}
              </small>
            </div>
            <div>
              <span>Aturan deduplikasi</span>
              <strong>1 hari · 1 ID</strong>
              <small>
                {{
                  overview.vision_ai?.enabled
                    ? `Wajah + Pakaian + Vision AI (${overview.vision_ai.model})`
                    : 'Wajah + Pola pakaian lokal (staf dikecualikan)'
                }}
              </small>
            </div>
          </section>

          <section class="camera-section">
            <div class="section-heading">
              <div>
                <h2>Kamera pengawas &amp; penghitung</h2>
                <p>Sumber: {{ overview.camera_id }} · Hijau = Karyawan, Biru = Customer Unik, Oranye = Observasi</p>
              </div>
              <span class="badge" :class="camera.running ? 'positive' : 'neutral'">
                {{ camera.running ? 'Sedang berjalan' : 'Belum berjalan' }}
              </span>
            </div>
            <div class="camera-layout">
              <div class="camera-stage">
                <canvas
                  v-show="camera.running && hasRenderedFrame"
                  ref="cameraCanvasRef"
                  aria-label="Pratinjau kamera teranotasi"
                ></canvas>
                <div v-if="!camera.running || !hasRenderedFrame" class="camera-empty">
                  <div class="viewfinder" aria-hidden="true"><span></span></div>
                  <strong>{{
                    camera.stopping
                      ? 'Menghentikan kamera…'
                      : camera.running
                        ? 'Menghubungkan kamera…'
                        : 'Kamera belum dinyalakan'
                  }}</strong>
                  <span>
                    {{
                      camera.stopping
                        ? 'Menunggu proses kamera selesai berhenti.'
                        : camera.running
                          ? 'Menunggu frame pertama dari proses AI.'
                          : 'Mulai pemrosesan untuk menghitung customer dan staf.'
                    }}
                  </span>
                </div>
              </div>
              <div class="camera-details">
                <span class="scope-meta">Kontrol penghitungan</span>
                <h3>Satu customer. <br />Satu hitungan harian.</h3>
                <p>
                  Pengunjung yang datang kembali pada hari yang sama dikenali melalui memori wajah dan pola pakaian
                  harian agar tidak terhitung ganda. Karyawan otomatis dikecualikan.
                </p>
                <div class="detail-row">
                  <span>Identitas kamera</span>
                  <strong>{{ overview.camera_id }}</strong>
                </div>
                <div class="detail-row">
                  <span>Performa inferensi</span>
                  <strong>{{ camera.running ? camera.fps + ' FPS' : '-' }}</strong>
                </div>
                <div class="detail-row">
                  <span>Sedang di layar</span>
                  <strong>
                    {{
                      camera.running
                        ? `${camera.live_customers || 0} customer, ${camera.live_staff || 0} staf`
                        : '-'
                    }}
                  </strong>
                </div>
                <p v-if="camera.error" class="form-error">{{ camera.error }}</p>
                <button class="button primary camera-action" :disabled="busy || camera.stopping" @click="toggleCamera">
                  {{ camera.stopping ? 'Menghentikan kamera…' : camera.running ? 'Hentikan kamera' : 'Mulai kamera & AI' }}
                  <span aria-hidden="true">{{ camera.running || camera.stopping ? '■' : '→' }}</span>
                </button>
              </div>
            </div>
          </section>

          <section class="list-section">
            <div class="section-heading">
              <div>
                <h2>Kehadiran karyawan hari ini</h2>
                <p>
                  {{ overview.staff_present_today }} dari {{ overview.employees }} karyawan aktif telah terdeteksi hari ini
                  dan tidak dimasukkan ke hitungan customer.
                </p>
              </div>
              <button class="text-button" @click="selectPage('Pegawai')">
                Kelola pegawai <span aria-hidden="true">→</span>
              </button>
            </div>
            <div v-if="!employees.length" class="empty-state">
              Belum ada pegawai terdaftar. Tambahkan pegawai melalui halaman Pegawai.
            </div>
            <div v-else class="data-table">
              <div class="table-row table-head">
                <span>Pegawai</span>
                <span>Status hari ini</span>
                <span>Terakhir terlihat</span>
                <span class="right">Referensi wajah</span>
              </div>
              <div
                v-for="emp in employees.slice(0, 5)"
                :key="emp.id"
                class="table-row clickable-row"
                @click="openEmployeeModal(emp)"
              >
                <strong>
                  {{ emp.name }}
                  <small>{{ emp.role }} · <span class="mono">{{ emp.code }}</span></small>
                </strong>
                <span>
                  <span class="badge" :class="emp.present_today ? 'positive' : 'neutral'">
                    {{ emp.present_today ? 'Hadir hari ini' : 'Belum terlihat' }}
                  </span>
                </span>
                <span>{{ emp.last_seen_today ? clockLabel(emp.last_seen_today) : '-' }}</span>
                <span class="right">{{ emp.embeddings }} vektor</span>
              </div>
            </div>
          </section>

          <section class="list-section">
            <div class="section-heading">
              <div>
                <h2>Customer unik terbaru hari ini</h2>
                <p>Setiap pengunjung baru dicatat satu kali per hari beserta jumlah kunjungan ulangnya.</p>
              </div>
              <button class="text-button" @click="selectPage('Customer')">
                Lihat semua ({{ overview.customers_today }}) <span aria-hidden="true">→</span>
              </button>
            </div>
            <div v-if="!visitors.length" class="empty-state">
              Belum ada customer terdeteksi hari ini. Jalankan kamera untuk mulai menghitung pengunjung.
            </div>
            <div v-else class="data-table">
              <div class="table-row table-head">
                <span>Identitas pengunjung</span>
                <span>Ciri pengenalan</span>
                <span>Pertama / Terakhir</span>
                <span class="right">Kunjungan hari ini</span>
              </div>
              <div v-for="vis in visitors.slice(0, 5)" :key="vis.id" class="table-row">
                <strong>
                  {{ vis.label }}
                  <small class="mono">{{ vis.code }}</small>
                </strong>
                <span>{{ vis.attributes }}</span>
                <span>{{ clockLabel(vis.first_seen_at) }} · {{ clockLabel(vis.last_seen_at) }}</span>
                <span class="right">
                  <span class="badge" :class="vis.visit_count > 1 ? 'warning' : 'neutral'">
                    {{ vis.visit_count }}× datang
                  </span>
                </span>
              </div>
            </div>
          </section>

          <section class="resource-summary" aria-label="Sumber daya perangkat">
            <div>
              <span>CPU host</span>
              <strong>{{ resources ? resources.cpu.percent + '%' : '-' }}</strong>
              <small>{{ resources?.cpu?.model || 'Menunggu perangkat' }}</small>
            </div>
            <div>
              <span>RAM host</span>
              <strong>{{ resources ? resources.memory.percent + '%' : '-' }}</strong>
              <small>
                {{
                  resources
                    ? `${gigabytes(resources.memory.used_bytes)} / ${gigabytes(resources.memory.total_bytes)}`
                    : 'Menunggu perangkat'
                }}
              </small>
            </div>
            <div>
              <span>Kamera</span>
              <strong>{{ camera.running ? `Indeks ${camera.active_config?.video_source ?? '-'}` : 'Nonaktif' }}</strong>
              <small>Detail tersedia di Pengaturan</small>
            </div>
          </section>
        </template>

        <template v-else-if="page === 'Customer'">
          <div class="tab-bar" role="tablist" aria-label="Tampilan data kunjungan">
            <button
              type="button"
              class="tab-button"
              :class="{ active: activityTab === 'visitors' }"
              @click="activityTab = 'visitors'"
            >
              Customer unik hari ini ({{ visitors.length }})
            </button>
            <button
              type="button"
              class="tab-button"
              :class="{ active: activityTab === 'events' }"
              @click="activityTab = 'events'"
            >
              Log deteksi kamera ({{ events.length }})
            </button>
          </div>

          <template v-if="activityTab === 'visitors'">
            <div class="list-toolbar">
              <button type="button" class="button secondary" @click="showReviewed = !showReviewed; loadReviewed()">{{ showReviewed ? 'Sembunyikan arsip salah' : 'Tampilkan arsip salah' }}</button>
            </div>
            <div class="list-toolbar">
              <div>
                <h2>Registri customer unik hari ini</h2>
                <p>
                  {{ filteredVisitors.length }} customer unik ditampilkan · pengunjung yang datang kembali di hari yang
                  sama digabung pada ID yang sama
                </p>
              </div>
              <input v-model="search" type="search" placeholder="Cari ID atau ciri pakaian" aria-label="Cari customer" />
            </div>
            <div v-if="showReviewed && reviewedVisitors.length" class="data-table">
              <div class="table-row table-head"><span>Arsip validasi</span><span>Status</span><span></span><span class="right">Aksi</span></div>
              <div v-for="vis in reviewedVisitors" :key="`review-${vis.id}`" class="table-row">
                <strong>{{ vis.label }}<small class="mono">{{ vis.code }}</small></strong><span>{{ vis.review_status }}</span><span></span>
                <span class="right"><button class="text-button" @click="restoreReviewed(vis)">Pulihkan</button><button class="text-button danger-text" @click="permanentlyDeleteReviewed(vis)">Hapus permanen</button></span>
              </div>
            </div>
            <div v-if="!showReviewed && !filteredVisitors.length" class="empty-state">
              {{ search ? 'Tidak ada customer yang cocok dengan pencarian.' : 'Belum ada customer tercatat hari ini.' }}
            </div>
            <div v-else class="data-table">
              <div class="table-row table-head">
                <span>Customer</span>
                <span>Basis pengenalan</span>
                <span>Pertama / Terakhir terlihat</span>
                <span class="right">Frekuensi hari ini</span>
              </div>
              <div
                v-for="vis in filteredVisitors"
                :key="vis.id"
                class="table-row customer-row-clickable"
                role="button"
                tabindex="0"
                :aria-label="`Lihat ${vis.label}, ${vis.photo_count || 0} foto`"
                @click="openCustomerPhotos(vis)"
                @keydown.enter="openCustomerPhotos(vis)"
                @keydown.space.prevent="openCustomerPhotos(vis)"
              >
                <strong>
                  {{ vis.label }}
                  <small class="mono">{{ vis.code }}</small>
                  <small>{{ vis.photo_count || 0 }} / 10 snapshot · klik untuk lihat</small>
                </strong>
                <span>{{ vis.attributes }}</span>
                <span>{{ timeLabel(vis.first_seen_at) }} → {{ clockLabel(vis.last_seen_at) }}</span>
                <span class="right">
                  <span class="badge" :class="vis.visit_count > 1 ? 'warning' : 'neutral'">
                    {{ vis.visit_count }}× kunjungan
                  </span>
                </span>
              </div>
            </div>
          </template>

          <template v-else>
            <div class="list-toolbar">
              <div class="filter-group">
                <button
                  type="button"
                  class="chip-button"
                  :class="{ selected: eventFilter === 'all' }"
                  @click="eventFilter = 'all'"
                >
                  Semua ({{ events.length }})
                </button>
                <button
                  type="button"
                  class="chip-button"
                  :class="{ selected: eventFilter === 'customer' }"
                  @click="eventFilter = 'customer'"
                >
                  Customer
                </button>
                <button
                  type="button"
                  class="chip-button"
                  :class="{ selected: eventFilter === 'staff' }"
                  @click="eventFilter = 'staff'"
                >
                  Karyawan
                </button>
              </div>
              <input v-model="search" type="search" placeholder="Cari nama, ID, atau kamera" aria-label="Cari aktivitas" />
            </div>
            <div v-if="!filteredEvents.length" class="empty-state">
              {{ search ? 'Tidak ada event yang cocok.' : 'Belum ada aktivitas tersimpan.' }}
            </div>
            <div v-else class="data-table">
              <div class="table-row table-head">
                <span>Subjek</span>
                <span>Kategori</span>
                <span>Waktu</span>
                <span class="right">Keyakinan</span>
              </div>
              <div v-for="event in filteredEvents" :key="event.id" class="table-row">
                <strong>
                  {{ event.label }}
                  <small>Kamera {{ event.camera_id }} · Track #{{ event.track_id }}</small>
                </strong>
                <span>
                  <span class="badge" :class="event.kind === 'staff' ? 'positive' : 'neutral'">
                    {{
                      event.kind === 'staff'
                        ? 'Karyawan'
                        : event.event_type === 'customer_return'
                          ? `Customer (Kunjungan ke-${event.visit_count || 2})`
                          : 'Customer baru'
                    }}
                  </span>
                </span>
                <span>{{ timeLabel(event.created_at) }}</span>
                <span class="right">{{ percent(event.confidence) }}</span>
              </div>
            </div>
          </template>
        </template>

        <template v-else-if="page === 'Pengaturan'">
          <div class="settings-layout">
            <section class="settings-panel">
              <h2>Konfigurasi kamera</h2>
              <p>Pengaturan tersimpan di database. Menyimpan saat kamera aktif mengulang worker, bukan server.</p>
              <form class="settings-form" @submit.prevent="saveSettings">
                <label for="source">Kamera / indeks perangkat</label>
                <input id="source" v-model.number="settings.video_source" type="number" min="0" max="8" required />
                <label for="level">Tingkat performa</label>
                <select id="level" v-model.number="settings.performance_level">
                  <option v-for="level in 5" :key="level" :value="level">
                    Level {{ level }}{{ level === 5 ? ' · paling ringan' : level === 1 ? ' · paling detail' : '' }}
                  </option>
                </select>
                <label for="width">Lebar tangkapan (opsional)</label>
                <input
                  id="width"
                  v-model="settings.camera_width"
                  type="number"
                  min="160"
                  max="1920"
                  placeholder="Ikuti preset"
                />
                <label for="height">Tinggi tangkapan (opsional)</label>
                <input
                  id="height"
                  v-model="settings.camera_height"
                  type="number"
                  min="120"
                  max="1080"
                  placeholder="Ikuti preset"
                />
                <label for="dwell">Durasi minimum customer (detik)</label>
                <input
                  id="dwell"
                  v-model.number="settings.customer_min_dwell_seconds"
                  type="number"
                  min="0.1"
                  max="10"
                  step="0.1"
                  required
                />
                <small>0,5 dtk direkomendasikan untuk orang yang hanya melintas.</small>
                <button class="button primary" type="submit" :disabled="busy">
                  {{ busy ? 'Menerapkan…' : 'Simpan & terapkan' }} <span aria-hidden="true">→</span>
                </button>
              </form>
              <p v-if="settingsNotice" class="info-note" role="status">{{ settingsNotice }}</p>
            </section>

            <section class="settings-panel satusehat-panel">
              <div class="satusehat-intro">
                <h2>Integrasi SatuSehat</h2>
                <p>Hubungkan fasilitas ke lingkungan SatuSehat. Mulai dari kredensial sandbox; gunakan production hanya setelah proses uji dan persetujuan selesai.</p>
                <ol class="satusehat-steps">
                  <li><span>1</span><div><strong>Daftar aplikasi</strong><small>Portal Developer SATUSEHAT → buat aplikasi, lalu pilih lingkungan yang sesuai.</small></div></li>
                  <li><span>2</span><div><strong>Masukkan ID fasilitas</strong><small>Organization ID berasal dari data organisasi/fasilitas yang terdaftar di SATUSEHAT.</small></div></li>
                  <li><span>3</span><div><strong>Simpan dan verifikasi</strong><small>Tes koneksi harus berhasil sebelum sinkronisasi Location atau Practitioner.</small></div></li>
                </ol>
                <a class="satusehat-doc-link" href="https://satusehat.kemkes.go.id/platform" target="_blank" rel="noopener noreferrer">Buka Portal Developer SATUSEHAT <span aria-hidden="true">↗</span></a>
                <p class="satusehat-security">Client Secret terenkripsi di database. Kunci enkripsi disimpan di <code>.env</code> dan tidak boleh hilang.</p>
              </div>
              <div class="satusehat-config">
                <p v-if="satusehat && !satusehat.encryption_key_configured" class="form-error" role="alert">
                  <code>SATUSEHAT_ENCRYPTION_KEY</code> belum diisi di <code>.env</code>. Isi kunci Fernet dan restart server sebelum menyimpan secret.
                </p>
                <form class="settings-form satusehat-form" autocomplete="off" @submit.prevent="saveSatuSehat">
                  <div class="ss-field ss-field-wide">
                    <label for="ss-env">Lingkungan SATUSEHAT</label>
                    <select id="ss-env" v-model="satusehatForm.env">
                      <option value="dev">Dev — sandbox</option>
                      <option value="staging">Staging — pra-produksi</option>
                      <option value="production">Production — layanan aktif</option>
                    </select>
                    <small>Harus sama dengan lingkungan tempat aplikasi dan kredensial dibuat.</small>
                    <code v-if="satusehat" class="ss-endpoint">{{ satusehat.environments[satusehatForm.env] }}</code>
                  </div>
                  <div class="ss-field">
                    <label for="ss-client-id">Client ID</label>
                    <input id="ss-client-id" v-model="satusehatForm.client_id" type="text" autocomplete="off" />
                    <small>Identitas aplikasi. Salin dari detail aplikasi di Portal Developer SATUSEHAT.</small>
                  </div>
                  <div class="ss-field">
                    <label for="ss-client-secret">Client Secret</label>
                    <input
                      id="ss-client-secret"
                      v-model="satusehatForm.client_secret"
                      type="password"
                      autocomplete="new-password"
                      :placeholder="satusehat?.client_secret_masked || 'Masukkan secret aplikasi'"
                    />
                    <small>Rahasia aplikasi dari portal yang sama. Kosongkan untuk mempertahankan secret tersimpan.</small>
                  </div>
                  <div class="ss-field">
                    <label for="ss-org">Organization ID</label>
                    <input id="ss-org" v-model="satusehatForm.organization_id" type="text" autocomplete="off" />
                    <small>ID organisasi/fasilitas kesehatan yang sudah terdaftar di SATUSEHAT, bukan Client ID.</small>
                  </div>
                  <div class="ss-field">
                    <label for="ss-location">Nama lokasi kamera</label>
                    <input id="ss-location" v-model="satusehatForm.location_name" type="text" placeholder="Mis. Ruang konsultasi 1" />
                    <small>Nama ruangan fisik; Location ID akan dibuat/ditemukan saat disinkronkan.</small>
                  </div>
                  <div class="ss-actions ss-field-wide">
                    <button class="button primary" type="submit" :disabled="Boolean(satusehatBusy) || (satusehat && !satusehat.encryption_key_configured)">
                      {{ satusehatBusy === 'save' ? 'Menyimpan…' : 'Simpan pengaturan' }}
                    </button>
                    <button class="button secondary" type="button" :disabled="Boolean(satusehatBusy) || !satusehat?.has_client_secret" @click="testSatuSehat">
                      {{ satusehatBusy === 'test' ? 'Menguji…' : 'Tes koneksi' }}
                    </button>
                    <button class="button secondary" type="button" :disabled="Boolean(satusehatBusy) || !satusehat?.has_client_secret" @click="syncSatuSehatLocation">
                      {{ satusehatBusy === 'location' ? 'Sinkron…' : 'Sinkronkan Location' }}
                    </button>
                  </div>
                </form>
                <p v-if="satusehatError" class="form-error" role="alert">{{ satusehatError }}</p>
                <p v-if="satusehatNotice" class="info-note" role="status">{{ satusehatNotice }}</p>
                <div v-if="satusehat" class="ss-connection-state">
                  <div><span>Tes koneksi terakhir</span><strong>{{ satusehat.last_test_at ? `${satusehat.last_test_ok ? 'Berhasil' : 'Gagal'} · ${timeLabel(satusehat.last_test_at)}` : 'Belum pernah diuji' }}</strong></div>
                  <div><span>Location ID</span><strong class="mono">{{ satusehat.location_id || 'Belum disinkronkan' }}</strong></div>
                  <p v-if="satusehat.last_test_message">{{ satusehat.last_test_message }}</p>
                </div>
              </div>
            </section>

            <section class="settings-panel">
              <h2>Perangkat &amp; mesin deduplikasi</h2>
              <p>Status pemakaian sumber daya host lokal dan metode pembeda customer vs karyawan.</p>
              <div class="detail-row">
                <span>Metode deduplikasi customer</span>
                <strong>ArcFace 512-D + HSV Torso Signature</strong>
              </div>
              <div class="detail-row">
                <span>Fallback Vision AI (Atribut/Seragam)</span>
                <strong>{{ overview.vision_ai?.model || 'Model belum diatur' }}</strong>
              </div>
              <div class="vision-status-row">
                <div>
                  <span class="vision-status" :class="`is-${overview.vision_ai?.status || 'checking'}`">
                    {{
                      ({ ready: 'Tersedia', checking: 'Sedang diuji', unavailable: 'Tidak tersedia', not_configured: 'Belum dikonfigurasi', disabled: 'Dinonaktifkan' })[overview.vision_ai?.status] || 'Belum diuji'
                    }}
                  </span>
                  <small>{{ overview.vision_ai?.message || 'Status koneksi belum diketahui.' }}</small>
                  <small v-if="overview.vision_ai?.last_checked_at">Tes terakhir: {{ timeLabel(overview.vision_ai.last_checked_at) }}</small>
                </div>
                <button class="button secondary" type="button" :disabled="visionAiBusy || overview.vision_ai?.status === 'checking'" @click="testVisionAi">
                  {{ visionAiBusy ? 'Menguji…' : 'Tes ulang' }}
                </button>
              </div>
              <div class="detail-row">
                <span>Prosesor</span>
                <strong>{{ resources?.cpu?.model || '-' }}</strong>
              </div>
              <div class="detail-row">
                <span>Arsitektur / inti</span>
                <strong>
                  {{
                    resources
                      ? `${resources.cpu.architecture} · ${resources.cpu.cores || '-'} inti / ${resources.cpu.threads || '-'} thread`
                      : '-'
                  }}
                </strong>
              </div>
              <div class="detail-row">
                <span>CPU host</span>
                <strong>{{ resources ? resources.cpu.percent + '%' : '-' }}</strong>
              </div>
              <div class="detail-row">
                <span>RAM host</span>
                <strong>
                  {{
                    resources
                      ? `${gigabytes(resources.memory.used_bytes)} / ${gigabytes(resources.memory.total_bytes)} · ${resources.memory.percent}%`
                      : '-'
                  }}
                </strong>
              </div>
              <div class="detail-row">
                <span>RAM proses AI + server</span>
                <strong>{{ resources ? gigabytes(resources.memory.process_bytes) : '-' }}</strong>
              </div>
              <div class="detail-row">
                <span>Kamera aktif</span>
                <strong>
                  {{ camera.running ? `Indeks ${camera.active_config?.video_source ?? settings.video_source}` : 'Nonaktif' }}
                </strong>
              </div>
              <p class="info-note">
                Foto wajah pegawai di db_karyawan diekstrak menjadi vektor 512-dimensi di PostgreSQL sehingga pencocokan
                berjalan langsung dari database.
              </p>
            </section>
          </div>
        </template>

        <template v-else>
          <div class="list-toolbar">
            <div>
              <h2>Direktori &amp; foto referensi pegawai</h2>
              <p>
                Klik pada baris pegawai untuk membuka galeri foto di folder <span class="mono">db_karyawan</span> atau
                menambah foto baru.
              </p>
            </div>
            <input v-model="search" type="search" placeholder="Cari nama, kode, atau folder" aria-label="Cari pegawai" />
          </div>
          <div v-if="!filteredEmployees.length" class="empty-state">
            {{ search ? 'Tidak ada pegawai yang cocok.' : 'Belum ada pegawai terdaftar. Klik "+ Tambah pegawai" di kanan atas.' }}
          </div>
          <div v-else class="data-table employee-table">
            <div class="table-row table-head">
              <span>Pegawai &amp; folder lokal</span>
              <span>Status hari ini</span>
              <span>Referensi wajah</span>
              <span class="right">Tindakan</span>
            </div>
            <div
              v-for="employee in filteredEmployees"
              :key="employee.id"
              class="table-row clickable-row"
              @click="openEmployeeModal(employee)"
            >
              <strong>
                {{ employee.name }}
                <small>
                  {{ employee.role }} · <span class="mono">{{ employee.code }}</span> ·
                  <span class="mono">{{ employee.folder }}</span>
                </small>
              </strong>
              <span>
                <span class="badge" :class="employee.present_today ? 'positive' : 'neutral'">
                  {{ employee.present_today ? `Hadir · ${clockLabel(employee.last_seen_today)}` : 'Belum terdeteksi' }}
                </span>
              </span>
              <span>{{ employee.embeddings }} vektor · {{ employee.photo_count ?? 0 }} foto</span>
              <span class="right">
                <button type="button" class="text-button" @click.stop="openEmployeeModal(employee)">
                  Foto &amp; edit →
                </button>
              </span>
            </div>
          </div>
          <p class="info-note">
            Setiap perubahan atau unggahan foto baru di halaman ini otomatis disimpan ke folder
            <span class="mono">db_karyawan/&lt;nama&gt;/</span> dan langsung diekstrak menjadi vektor 512-D ke PostgreSQL.
          </p>
        </template>

        <footer class="sheet-footer">
          <span>StaffLens AI · Penghitung Customer Unik &amp; Kehadiran Staf</span>
          <span>{{ lastUpdated ? 'Diperbarui ' + timeLabel(lastUpdated) : 'Menunggu data' }}</span>
        </footer>
      </main>
    </div>

    <div v-if="customerPhotoModal.open" class="modal-backdrop" @click.self="closeCustomerPhotos">
      <div class="modal-dialog customer-photo-modal" role="dialog" aria-modal="true" aria-labelledby="customer-photo-title">
        <div class="modal-header">
          <div>
            <span class="scope-meta modal-eyebrow">GALERI SNAPSHOT LOKAL · MAKS. 10</span>
            <h2 id="customer-photo-title">{{ customerPhotoModal.customer?.label }} · {{ customerPhotoModal.customer?.code }}</h2>
            <p>Snapshot crop customer terpilih berdasarkan ketajaman gambar. Tersimpan lokal di db_customer/.</p>
          </div>
          <div class="modal-actions">
            <button type="button" class="button danger-button" @click="openCustomerReview(customerPhotoModal.customer)">Tandai salah / Hapus</button>
            <button type="button" class="button secondary" @click="closeCustomerPhotos">Tutup</button>
          </div>
        </div>
        <div class="customer-photo-grid">
          <p v-if="customerPhotoModal.loading" class="empty-state">Memuat snapshot…</p>
          <p v-else-if="customerPhotoModal.error" class="form-error">{{ customerPhotoModal.error }}</p>
          <p v-else-if="!customerPhotoModal.photos.length" class="empty-state">Belum ada snapshot tersimpan. Snapshot diambil saat kamera berjalan.</p>
          <img
            v-for="photo in customerPhotoModal.photos"
            :key="photo.name"
            :src="photo.url"
            :alt="`Snapshot ${customerPhotoModal.customer?.label}`"
            loading="lazy"
          />
        </div>
      </div>
    </div>

    <div v-if="customerReview.open" class="modal-backdrop" @click.self="customerReview.open = false">
      <div class="modal-dialog" role="dialog" aria-modal="true">
        <div class="modal-header"><div><span class="scope-meta modal-eyebrow">VALIDASI CUSTOMER</span><h2>Tandai salah / Hapus</h2><p>Pilih alasan. Feedback tersimpan di PostgreSQL.</p></div><button class="button secondary" @click="customerReview.open = false">Tutup</button></div>
        <div class="review-reasons">
          <button v-for="item in reviewReasons" :key="item[0]" type="button" class="button" :class="customerReview.reason === item[0] ? 'primary' : 'secondary'" @click="customerReview.reason = item[0]">{{ item[1] }}</button>
        </div>
        <textarea v-if="customerReview.reason === 'other'" v-model="customerReview.note" rows="3" placeholder="Catatan opsional"></textarea>
        <p v-if="customerReview.error" class="form-error">{{ customerReview.error }}</p>
        <button type="button" class="button danger-button" :disabled="!customerReview.reason || customerReview.busy" @click="submitCustomerReview">{{ customerReview.busy ? 'Menyimpan…' : 'Konfirmasi hapus' }}</button>
      </div>
    </div>

    <!-- Modal Tambah / Edit Pegawai & Galeri Foto db_karyawan -->
    <div v-if="showEmployeeModal" class="modal-backdrop" @click.self="closeEmployeeModal">
      <div class="modal-dialog employee-modal" role="dialog" aria-modal="true" aria-labelledby="modal-emp-title">
        <div class="modal-header">
          <div>
            <span class="scope-meta modal-eyebrow">
              {{ modalEmployee.id ? `DATA PEGAWAI · #${modalEmployee.id}` : 'REGISTRASI PEGAWAI BARU' }}
            </span>
            <h2 id="modal-emp-title">
              {{ modalEmployee.id ? `Kelola ${modalEmployee.name}` : 'Tambah Pegawai & Foto Wajah' }}
            </h2>
            <p>
              {{ modalEmployee.id ? 'Perbarui data pegawai dan foto referensi.' : 'Isi data pegawai, lalu tambahkan foto wajah untuk pengenalan kamera.' }}
            </p>
          </div>
          <button type="button" class="button secondary" :disabled="modalSaving" @click="closeEmployeeModal">
            Tutup
          </button>
        </div>

        <form class="modal-body employee-modal-body" @submit.prevent="saveEmployeeModal(false)">
          <div class="modal-key" aria-label="Penanda isian">
            <span><b class="required-mark">*</b> Wajib diisi</span>
            <span>Opsional boleh dikosongkan</span>
          </div>
          <section class="employee-fields-section">
            <div class="employee-section-title"><span>01</span><div><h3>Data pegawai</h3><p>Informasi dasar untuk daftar karyawan.</p></div></div>
          <div class="modal-form-grid">
            <div>
              <label for="emp-code">Kode pegawai <b class="required-mark">*</b></label>
              <input
                id="emp-code"
                v-model="modalEmployee.code"
                type="text"
                placeholder="EMP-003"
                required
              />
            </div>
            <div>
              <label for="emp-name">Nama lengkap <b class="required-mark">*</b></label>
              <input
                id="emp-name"
                v-model="modalEmployee.name"
                type="text"
                placeholder="Mis. Siti"
                required
              />
            </div>
            <div>
              <label for="emp-role">Peran / jabatan <b class="required-mark">*</b></label>
              <input
                id="emp-role"
                v-model="modalEmployee.role"
                type="text"
                placeholder="Karyawan"
                required
              />
            </div>
          </div>
          </section>

          <div class="photo-section employee-section">
            <div class="photo-section-head">
              <div>
                <h3>Data SatuSehat <span class="optional-mark">Opsional</span></h3>
                <p>
                  NIK dienkripsi di database dan hanya dipakai untuk mencari Practitioner. Tidak pernah ditampilkan
                  utuh.
                </p>
              </div>
              <button
                v-if="modalEmployee.id"
                type="button"
                class="button secondary"
                :disabled="modalSaving || !modalEmployee.has_nik"
                @click="syncEmployeeSatuSehat"
              >
                Sinkron ke SatuSehat
              </button>
            </div>
            <div class="modal-form-grid">
              <div>
                <label for="emp-nik">NIK <span class="optional-mark">Opsional · 16 digit</span></label>
                <input
                  id="emp-nik"
                  v-model="modalEmployee.nik"
                  type="text"
                  inputmode="numeric"
                  maxlength="16"
                  autocomplete="off"
                  :placeholder="modalEmployee.nik_masked || 'Belum diisi'"
                />
              </div>
              <div>
                <label for="emp-practitioner">ID Practitioner <span class="optional-mark">Otomatis</span></label>
                <input
                  id="emp-practitioner"
                  type="text"
                  readonly
                  :value="modalEmployee.satusehat_practitioner_id || '-'"
                />
              </div>
            </div>
            <label v-if="modalEmployee.has_nik" class="checkbox-row">
              <input v-model="modalEmployee.nik_clear" type="checkbox" />
              <span>Hapus NIK tersimpan saat menyimpan</span>
            </label>
          </div>

          <label class="checkbox-row">
            <input v-model="modalEmployee.active" type="checkbox" />
            <span>Pegawai aktif (dikenali kamera &amp; dikecualikan dari hitungan customer)</span>
          </label>

          <div v-if="cameraCaptureOpen" class="camera-capture-panel">
            <video ref="cameraVideoRef" autoplay playsinline muted></video>
            <div class="camera-capture-actions">
              <button type="button" class="button primary" @click="capturePhoto">Jepret foto</button>
              <button type="button" class="button secondary" @click="stopPhotoCamera">Tutup kamera</button>
            </div>
            <small>Pastikan izin kamera diberikan. Browser perlu HTTPS atau localhost.</small>
          </div>
          <div class="photo-section employee-section">
            <div class="photo-section-head">
              <div>
                <h3>Foto referensi <b v-if="!modalEmployee.id" class="required-mark">*</b><span v-else class="optional-mark">Opsional</span></h3>
                <p>
                  {{ modalEmployee.id ? 'Tambahkan foto baru atau tandai foto lama yang ingin dihapus.' : 'Minimal 1 foto wajib. Gunakan wajah terang, satu orang per foto; disarankan 3–5 foto.' }}
                </p>
              </div>
              <div class="photo-source-actions">
                <button type="button" class="button secondary" @click="startPhotoCamera">📷 Ambil foto</button>
                <label class="button secondary upload-trigger">
                  + Pilih foto (.jpg/.png)
                <input
                  type="file"
                  accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                  multiple
                  hidden
                  @change="onSelectPhotoFiles"
                />
                </label>
              </div>
            </div>

            <div v-if="modalLoading" class="empty-state">Memuat foto dari folder db_karyawan…</div>
            <div
              v-else-if="!modalEmployee.photos?.length && !pendingUploads.length"
              class="empty-state"
            >
              Belum ada foto di folder ini. Klik "+ Pilih foto (.jpg/.png)" untuk mengunggah foto wajah pegawai.
            </div>
            <div v-else class="photo-grid">
              <div
                v-for="photo in modalEmployee.photos"
                :key="photo.name"
                class="photo-card"
                :class="{ marked: photosToDelete.includes(photo.name) }"
              >
                <div class="photo-thumb">
                  <img :src="photo.url" :alt="photo.name" loading="lazy" />
                </div>
                <div class="photo-meta">
                  <div>
                    <strong class="mono">{{ photo.name }}</strong>
                    <small>{{ kilobytes(photo.size_bytes) }} · Tersimpan</small>
                  </div>
                  <button
                    type="button"
                    class="text-button"
                    @click="toggleDeleteExistingPhoto(photo.name)"
                  >
                    {{ photosToDelete.includes(photo.name) ? 'Batal hapus' : 'Hapus' }}
                  </button>
                </div>
              </div>

              <div
                v-for="(item, idx) in pendingUploads"
                :key="'new-' + idx"
                class="photo-card pending"
              >
                <div class="photo-thumb">
                  <img :src="item.data_url" :alt="item.filename" />
                </div>
                <div class="photo-meta">
                  <div>
                    <strong class="mono">{{ item.filename }}</strong>
                    <small>{{ kilobytes(item.size_bytes) }} · Baru</small>
                  </div>
                  <button type="button" class="text-button" @click="removePendingUpload(idx)">
                    Batal
                  </button>
                </div>
              </div>
            </div>
          </div>

          <p v-if="modalError" class="form-error" role="alert">{{ modalError }}</p>
          <p v-if="modalNotice" class="info-note modal-success" role="status">{{ modalNotice }}</p>

          <div class="modal-footer employee-modal-footer">
            <button v-if="modalEmployee.id" type="button" class="button danger-button" :disabled="modalSaving || employeeDeleting" @click="deleteEmployee">
              {{ employeeDeleting ? 'Menghapus…' : 'Hapus pegawai' }}
            </button>
            <span class="scope-meta employee-save-summary">
              {{ modalEmployee.embeddings || 0 }} vektor · {{ pendingUploads.length }} foto baru
            </span>
            <div class="modal-actions">
              <button
                v-if="modalEmployee.id && modalEmployee.photos?.length"
                type="button"
                class="button secondary"
                :disabled="modalSaving"
                @click="saveEmployeeModal(true)"
              >
                Ekstrak ulang vektor
              </button>
              <button type="submit" class="button primary" :disabled="modalSaving || employeeDeleting">
                {{
                  modalSaving
                    ? 'Menyimpan & mengekstrak wajah…'
                    : modalEmployee.id
                      ? 'Simpan perubahan →'
                      : 'Simpan & daftarkan pegawai →'
                }}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>
