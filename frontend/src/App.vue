<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import './style.css'

const authenticated = ref(false)
const checking = ref(true)
const password = ref('')
const loginError = ref('')
const busy = ref(false)
const page = ref('Ringkasan')
const activityTab = ref('visitors')
const eventFilter = ref('all')
const settings = ref({ video_source: 0, performance_level: 5, camera_width: null, camera_height: null })
const resources = ref(null)
const settingsNotice = ref('')
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
  vision_ai: { enabled: false, model: null, configured: false },
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
const frameUrl = ref('')
let streamId = 0
const search = ref('')
let timer
let resourcesTimer

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

const navItems = ['Ringkasan', 'Customer', 'Pegawai', 'Pengaturan']

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, { credentials: 'same-origin', cache: 'no-store', ...options })
  if (response.status === 401 && path !== '/auth/login') {
    authenticated.value = false
    frameUrl.value = ''
    stopPolling()
    throw new Error('Sesi berakhir. Masuk kembali.')
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `Permintaan gagal (${response.status})`)
  }
  return response.json()
}

async function refresh() {
  if (!authenticated.value) return
  try {
    const [summary, visitorData, eventData, employeeData, cameraData] = await Promise.all([
      api('/overview'),
      api('/visitors?limit=100'),
      api('/events?limit=60'),
      api('/employees'),
      api('/camera/status'),
    ])
    overview.value = summary
    visitors.value = visitorData.items || []
    events.value = eventData.items || []
    employees.value = employeeData.items || []
    camera.value = cameraData
    if (!cameraData.running) frameUrl.value = ''
    else if (cameraData.has_frame && !frameUrl.value) frameUrl.value = `/api/camera/stream?session=${++streamId}`
    lastUpdated.value = new Date()
    dataError.value = ''
  } catch (error) {
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
  }
  try {
    const detail = await api(`/employees/${emp.id}`)
    modalEmployee.value = detail
  } catch (error) {
    modalError.value = error.message
  } finally {
    modalLoading.value = false
  }
}

function closeEmployeeModal() {
  if (modalSaving.value) return
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
        new_photos: pendingUploads.value.map(item => ({
          filename: item.filename,
          data_url: item.data_url,
        })),
        delete_photos: photosToDelete.value,
        reprocess_embeddings: Boolean(reprocessOnly),
      }),
    })
    modalEmployee.value = result
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

function stopPolling() {
  clearInterval(timer)
  clearInterval(resourcesTimer)
  timer = undefined
  resourcesTimer = undefined
}

function startPolling() {
  stopPolling()
  refresh()
  refreshResources()
  loadSettings()
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
  frameUrl.value = ''
  authenticated.value = false
  stopPolling()
  try {
    await api('/auth/logout', { method: 'POST' })
  } catch (_) {
    /* clear local view regardless */
  }
}

async function toggleCamera() {
  busy.value = true
  try {
    if (camera.value.running) frameUrl.value = ''
    await api(camera.value.running ? '/camera/stop' : '/camera/start', { method: 'POST' })
    await refresh()
  } catch (error) {
    dataError.value = error.message
  } finally {
    busy.value = false
  }
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
  frameUrl.value = ''
})
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
          @click="page = item; search = ''"
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
          @click="page = item; search = ''"
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
            <button class="button secondary" @click="refresh">
              Perbarui data <span aria-hidden="true">↗</span>
            </button>
          </div>
        </div>

        <p v-if="dataError" class="error-notice" role="alert">
          {{ dataError }}
          <button class="text-button" @click="refresh">Coba lagi</button>
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
                <img
                  v-if="frameUrl"
                  :src="frameUrl"
                  alt="Pratinjau kamera teranotasi"
                  @error="frameUrl = ''"
                />
                <div v-else class="camera-empty">
                  <div class="viewfinder" aria-hidden="true"><span></span></div>
                  <strong>{{ camera.running ? 'Menghubungkan kamera…' : 'Kamera belum dinyalakan' }}</strong>
                  <span>
                    {{
                      camera.running
                        ? 'Menunggu frame pertama dari proses AI.'
                        : 'Mulai pemrosesan untuk menghitung customer dan staf.'
                    }}
                  </span>
                </div>
                <span class="camera-overlay">
                  {{
                    camera.running
                      ? `LIVE · ${camera.fps} FPS · DI LAYAR: ${camera.live_customers || 0} CUST, ${camera.live_staff || 0} STAF · HARI INI: ${overview.customers_today} CUST`
                      : 'OFFLINE'
                  }}
                </span>
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
                <button class="button primary camera-action" :disabled="busy" @click="toggleCamera">
                  {{ camera.running ? 'Hentikan kamera' : 'Mulai kamera & AI' }}
                  <span aria-hidden="true">{{ camera.running ? '■' : '→' }}</span>
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
              <button class="text-button" @click="page = 'Pegawai'">
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
              <button class="text-button" @click="page = 'Customer'">
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
              <div>
                <h2>Registri customer unik hari ini</h2>
                <p>
                  {{ filteredVisitors.length }} customer unik ditampilkan · pengunjung yang datang kembali di hari yang
                  sama digabung pada ID yang sama
                </p>
              </div>
              <input v-model="search" type="search" placeholder="Cari ID atau ciri pakaian" aria-label="Cari customer" />
            </div>
            <div v-if="!filteredVisitors.length" class="empty-state">
              {{ search ? 'Tidak ada customer yang cocok dengan pencarian.' : 'Belum ada customer tercatat hari ini.' }}
            </div>
            <div v-else class="data-table">
              <div class="table-row table-head">
                <span>Customer</span>
                <span>Basis pengenalan</span>
                <span>Pertama / Terakhir terlihat</span>
                <span class="right">Frekuensi hari ini</span>
              </div>
              <div v-for="vis in filteredVisitors" :key="vis.id" class="table-row">
                <strong>
                  {{ vis.label }}
                  <small class="mono">{{ vis.code }}</small>
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
                <button class="button primary" type="submit" :disabled="busy">
                  {{ busy ? 'Menerapkan…' : 'Simpan & terapkan' }} <span aria-hidden="true">→</span>
                </button>
              </form>
              <p v-if="settingsNotice" class="info-note" role="status">{{ settingsNotice }}</p>
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
                <strong>
                  {{
                    overview.vision_ai?.enabled
                      ? `Aktif (${overview.vision_ai.model})`
                      : 'Siap (Isi VISION_AI_API_KEY di .env bila diperlukan)'
                  }}
                </strong>
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

    <!-- Modal Tambah / Edit Pegawai & Galeri Foto db_karyawan -->
    <div v-if="showEmployeeModal" class="modal-backdrop" @click.self="closeEmployeeModal">
      <div class="modal-dialog" role="dialog" aria-modal="true" aria-labelledby="modal-emp-title">
        <div class="modal-header">
          <div>
            <span class="scope-meta">
              {{ modalEmployee.id ? `ID Pegawai #${modalEmployee.id}` : 'Registrasi Pegawai Baru' }}
            </span>
            <h2 id="modal-emp-title">
              {{ modalEmployee.id ? `Kelola ${modalEmployee.name}` : 'Tambah Pegawai & Foto Wajah' }}
            </h2>
            <p>
              Folder direktori:
              <span class="mono">{{ modalEmployee.folder || folderSlugPreview }}</span>
            </p>
          </div>
          <button type="button" class="button secondary" :disabled="modalSaving" @click="closeEmployeeModal">
            Tutup
          </button>
        </div>

        <form class="modal-body" @submit.prevent="saveEmployeeModal(false)">
          <div class="modal-form-grid">
            <div>
              <label for="emp-code">Kode pegawai</label>
              <input
                id="emp-code"
                v-model="modalEmployee.code"
                type="text"
                placeholder="EMP-003"
                required
              />
            </div>
            <div>
              <label for="emp-name">Nama lengkap</label>
              <input
                id="emp-name"
                v-model="modalEmployee.name"
                type="text"
                placeholder="Mis. Siti"
                required
              />
            </div>
            <div>
              <label for="emp-role">Peran / jabatan</label>
              <input
                id="emp-role"
                v-model="modalEmployee.role"
                type="text"
                placeholder="Karyawan"
                required
              />
            </div>
          </div>

          <label class="checkbox-row">
            <input v-model="modalEmployee.active" type="checkbox" />
            <span>Pegawai aktif (dikenali kamera &amp; dikecualikan dari hitungan customer)</span>
          </label>

          <div class="photo-section">
            <div class="photo-section-head">
              <div>
                <h3>Galeri foto di {{ modalEmployee.folder || folderSlugPreview }}</h3>
                <p>
                  Gunakan 3–5 foto wajah terang (tepat 1 wajah per foto). Saat disimpan, InsightFace otomatis membuat
                  vektor 512-D ke PostgreSQL.
                </p>
              </div>
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

          <div class="modal-footer">
            <span class="scope-meta">
              {{ modalEmployee.embeddings || 0 }} vektor aktif di DB · {{ pendingUploads.length }} foto baru siap diunggah
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
              <button type="submit" class="button primary" :disabled="modalSaving">
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
