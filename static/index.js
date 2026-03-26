const tbody = document.querySelector("#holdings tbody");
const totalEl = document.getElementById("total");
const errorEl = document.getElementById("error");
const loadingEl = document.getElementById("loading");
const subtotalBody = document.querySelector("#subtotalTable tbody");
const pieChartEl = document.getElementById("pieChart");
const pieLegendEl = document.getElementById("pieLegend");
const authUserEl = document.getElementById("authUser");
const authActionsEl = document.getElementById("authActions");
const holdingsSectionEl = document.querySelector(".layout > section.card");
const sideCardEl = document.querySelector(".side-card");
const subtotalWrapEl = document.querySelector(".subtotal-table-wrap");
const saveAllBtn = document.getElementById("saveAll");
const refreshBtn = document.getElementById("refresh");
const importBtn = document.getElementById("importCsvText");
const toastEl = document.getElementById("toast");
const dialogModalEl = document.getElementById("dialogModal");
const dialogTitleEl = document.getElementById("dialogTitle");
const dialogMessageEl = document.getElementById("dialogMessage");
const dialogConfirmBtnEl = document.getElementById("dialogConfirmBtn");
const dialogCancelBtnEl = document.getElementById("dialogCancelBtn");
let currentUser = null;
const SAMPLE_CSV_TEXT =
  "ticker,quantity\n005930,10\n005380,10\n000660,10\n035420,10\n064400,10\n018260,10\nna,600000";
const CSV_HEADER_TEXT = "ticker,quantity";
const LOGIN_REQUIRED_MESSAGE = "로그인이 필요합니다. Google 계정으로 로그인 후 이용하세요.";
let dialogResolver = null;
let dialogCloseOnBackdrop = true;

function hideDialogModal(result = false) {
  if (!dialogModalEl) return;
  dialogModalEl.classList.remove("show");
  if (dialogResolver) {
    dialogResolver(result);
    dialogResolver = null;
  }
}

function showDialog({
  title = "안내",
  message = "",
  confirmText = "확인",
  cancelText = "취소",
  showCancel = true,
  closeOnBackdrop = true,
} = {}) {
  if (!dialogModalEl) {
    return Promise.resolve(window.confirm(message));
  }

  dialogTitleEl.textContent = title;
  dialogMessageEl.textContent = message;
  dialogConfirmBtnEl.textContent = confirmText;
  dialogCancelBtnEl.textContent = cancelText;
  dialogCancelBtnEl.style.display = showCancel ? "" : "none";
  dialogCloseOnBackdrop = closeOnBackdrop;
  dialogModalEl.classList.add("show");

  return new Promise((resolve) => {
    dialogResolver = resolve;
  });
}

function syncSideCardHeight() {
  if (!holdingsSectionEl || !sideCardEl || !subtotalWrapEl) return;
  if (window.innerWidth <= 900) {
    holdingsSectionEl.style.height = "";
    sideCardEl.style.height = "";
    subtotalWrapEl.style.height = "";
    return;
  }

  window.requestAnimationFrame(() => {
    holdingsSectionEl.style.height = "";
    sideCardEl.style.height = "";
    subtotalWrapEl.style.height = "";
    const leftHeight = holdingsSectionEl.getBoundingClientRect().height;
    const rightHeight = sideCardEl.getBoundingClientRect().height;
    const targetHeight = Math.max(leftHeight, rightHeight);
    if (targetHeight > 0) {
      holdingsSectionEl.style.height = `${Math.round(targetHeight)}px`;
      sideCardEl.style.height = `${Math.round(targetHeight)}px`;
    }
  });
}

function createRow(data = {}) {
  const tr = document.createElement("tr");
  tr.dataset.id = data.id || "";
  tr.innerHTML = `
  <td data-label="종목유형"><input type="text" placeholder="ETF/주식/예수금" value="${data.asset_type || "주식"}" /></td>
  <td data-label="티커"><input type="text" placeholder="예: 005930, AAPL" value="${data.ticker || ""}" /></td>
  <td data-label="종목명" class="name">${data.name || "-"}</td>
  <td data-label="현재가" class="price">${data.price ? formatMoney(data.price, data.currency === "KRW" ? "₩" : data.currency) : "-"}</td>
  <td data-label="수량"><input type="number" min="0" step="1" value="${data.quantity ?? 0}" /></td>
  <td data-label="보유금액" class="amount">${data.amount ? formatMoney(data.amount, data.currency === "KRW" ? "₩" : data.currency) : "-"}</td>
  <td data-label="삭제"><button class="ghost remove">삭제</button></td>
  `;
  tr.querySelector(".remove").addEventListener("click", () =>
    handleDelete(tr),
  );
  return tr;
}

function formatMoney(value, currency = "") {
  if (value === null || value === undefined || Number.isNaN(value))
    return "-";
  const formatter = new Intl.NumberFormat("ko-KR", {
    maximumFractionDigits: 2,
  });
  return `${currency ? currency + " " : ""}${formatter.format(value)}`;
}

function setLoading(isLoading, message = "조회 중...") {
  loadingEl.style.display = isLoading ? "block" : "none";
  loadingEl.textContent = message;
  saveAllBtn.disabled = isLoading;
  refreshBtn.disabled = isLoading;
  importBtn.disabled = isLoading;
}

function showLoginRequired() {
  errorEl.textContent = LOGIN_REQUIRED_MESSAGE;
  showToast(LOGIN_REQUIRED_MESSAGE);
}

function updateCsvPlaceholder(hasExistingHoldings) {
  const csvTextEl = document.getElementById("csvText");
  if (!csvTextEl) return;
  csvTextEl.placeholder = hasExistingHoldings ? CSV_HEADER_TEXT : SAMPLE_CSV_TEXT;
}

function parseAmount(text) {
  const cleaned = text.replace(/[₩,\\s]/g, "");
  if (!cleaned) return null;
  const value = Number(cleaned);
  return Number.isNaN(value) ? null : value;
}

function updateSubtotals(entries) {
  const totals = new Map();
  const qtyTotals = new Map();
  let grandTotal = 0;
  
  entries.forEach((row) => {
    const amount = Number(row.amount || 0);
    if (isNaN(amount) || amount <= 0) return;
    
    grandTotal += amount;
    const key = row.name || "-";
    totals.set(key, (totals.get(key) || 0) + amount);
    const qty = Number(row.quantity || 0);
    qtyTotals.set(key, (qtyTotals.get(key) || 0) + qty);
  });

  const summaryTotalEl = document.getElementById("summaryTotal");
  if (summaryTotalEl) {
    summaryTotalEl.textContent = formatMoney(grandTotal, "₩");
  }

  const list = Array.from(totals.entries()).sort((a, b) => b[1] - a[1]);
  subtotalBody.innerHTML = "";
  list.forEach(([name, amount]) => {
    const ratio = grandTotal > 0 ? (amount / grandTotal) * 100 : 0;
    const qtySum = qtyTotals.get(name) || 0;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${name}</td>
      <td>${new Intl.NumberFormat("ko-KR").format(qtySum)}</td>
      <td>${formatMoney(amount, "₩")}</td>
      <td>${ratio.toFixed(1)}%</td>
    `;
    subtotalBody.appendChild(tr);
  });

  renderPie(list, grandTotal);
  syncSideCardHeight();
}

function updateSubtotalsFromDom() {
  const rows = Array.from(tbody.querySelectorAll("tr"));
  const entries = rows.map((row) => {
    const name = row.querySelector(".name")?.textContent || "-";
    const amountText = row.querySelector(".amount")?.textContent || "";
    const amount = parseAmount(amountText);
    const qtyValue = Number(
      row.querySelector("td:nth-child(5) input")?.value || 0,
    );
    return { name, amount, quantity: qtyValue };
  });
  updateSubtotals(entries);
}

function renderPie(entries, total) {
  pieChartEl.innerHTML = "";
  pieLegendEl.innerHTML = "";
  if (!entries.length || total === 0) {
    pieChartEl.textContent = "데이터 없음";
    return;
  }

  const size = 280;
  const radius = 125;
  const center = size / 2;
  const colors = [
    "#0f766e",
    "#3f4e4f",
    "#a66e4a",
    "#7b8a70",
    "#c37b6a",
    "#5c6b73",
    "#9a7d6a",
    "#6d8b74",
  ];
  const svgDefs = `
    <defs>
      <radialGradient id="pieShade" cx="35%" cy="35%" r="70%">
        <stop offset="0%" stop-color="#ffffff" stop-opacity="0.6" />
        <stop offset="65%" stop-color="#ffffff" stop-opacity="0.0" />
        <stop offset="100%" stop-color="#000000" stop-opacity="0.08" />
      </radialGradient>
      <filter id="pieShadow" x="-30%" y="-30%" width="160%" height="160%">
        <feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.18" />
      </filter>
    </defs>
  `;

  let startAngle = -Math.PI / 2;
  const svgParts = [];
  const labelParts = [];

  entries.forEach(([name, amount], idx) => {
    const angle = (amount / total) * Math.PI * 2;
    const ratio = total > 0 ? (amount / total) * 100 : 0;
    const endAngle = startAngle + angle;
    const largeArc = angle > Math.PI ? 1 : 0;
    const x1 = center + radius * Math.cos(startAngle);
    const y1 = center + radius * Math.sin(startAngle);
    const x2 = center + radius * Math.cos(endAngle);
    const y2 = center + radius * Math.sin(endAngle);
    const color = colors[idx % colors.length];

    const midAngle = startAngle + angle / 2;
    const explode = 4;
    const dx = Math.cos(midAngle) * explode;
    const dy = Math.sin(midAngle) * explode;

    svgParts.push(
      `<path d="M ${center} ${center} L ${x1} ${y1} A ${radius} ${radius} 0 ${largeArc} 1 ${x2} ${y2} Z" fill="${color}" stroke="#f6f3ee" stroke-width="2" filter="url(#pieShadow)" transform="translate(${dx}, ${dy})"></path>`,
    );

    if (ratio >= 3) {
      const lx = center + radius * 0.6 * Math.cos(midAngle) + dx;
      const ly = center + radius * 0.6 * Math.sin(midAngle) + dy;
      const subtotalText = idx < 3 ? formatMoney(amount, "₩") : "";
      if (subtotalText) {
        labelParts.push(
          `<text x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="bold" fill="#1b1a18">
            <tspan x="${lx}" dy="-7">${ratio.toFixed(1)}%</tspan>
            <tspan x="${lx}" dy="14">${subtotalText}</tspan>
          </text>`
        );
      } else {
        labelParts.push(
          `<text x="${lx}" y="${ly}" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="bold" fill="#1b1a18">${ratio.toFixed(1)}%</text>`
        );
      }
    }

    const legendItem = document.createElement("div");
    legendItem.className = "legend-item";
    legendItem.innerHTML = `<span class="legend-swatch" style="background:${color}"></span>${name} (${ratio.toFixed(1)}%)`;
    pieLegendEl.appendChild(legendItem);

    startAngle = endAngle;
  });

  pieChartEl.innerHTML = `
    <svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}">
      ${svgDefs}
      ${svgParts.join("")}
      <circle cx="${center}" cy="${center}" r="${radius}" fill="url(#pieShade)" />
      ${labelParts.join("")}
    </svg>
  `;
}

async function fetchQuote(ticker) {
  const res = await fetch(
    `/api/quote?ticker=${encodeURIComponent(ticker)}`,
  );
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const msg = data.detail || "조회 실패";
    throw new Error(`${ticker}: ${msg}`);
  }
  return res.json();
}

async function loadHoldings() {
  errorEl.textContent = "";
  setLoading(true, "조회 중... (0/0)");
  const res = await fetch(`/api/holdings_raw`);
  if (res.status === 401) {
    errorEl.textContent = LOGIN_REQUIRED_MESSAGE;
    updateCsvPlaceholder(false);
    setLoading(false);
    return;
  }
  if (!res.ok) {
    errorEl.textContent = "목록을 불러오지 못했습니다.";
    setLoading(false);
    return;
  }
  const data = await res.json();
  updateCsvPlaceholder(data.length > 0);
  tbody.innerHTML = "";
  let total = 0;
  const failedTickers = [];
  data.forEach((row) => {
    tbody.appendChild(createRow(row));
  });

  for (let i = 0; i < data.length; i++) {
    const row = data[i];
    setLoading(
      true,
      `조회 중... (${i + 1}/${data.length}) ${row.ticker}`,
    );
    const tr = tbody.children[i];
    const nameEl = tr.querySelector(".name");
    const priceEl = tr.querySelector(".price");
    const amountEl = tr.querySelector(".amount");
    try {
      const resQuote = await fetch(
        `/api/quote_krw?ticker=${encodeURIComponent(row.ticker)}`,
      );
      if (!resQuote.ok) {
        const data = await resQuote.json().catch(() => ({}));
        const detail = data.detail || "조회 실패";
        throw new Error(detail);
      }
      const quote = await resQuote.json();
      nameEl.textContent = quote.name;
      priceEl.textContent = formatMoney(quote.price, "₩");
      const amount = quote.price * row.quantity;
      amountEl.textContent = formatMoney(amount, "₩");
      total += amount;
      
      // 데이터 객체 업데이트 (정렬 후 재렌더링 시 필요)
      row.name = quote.name;
      row.price = quote.price;
      row.amount = amount;
      row.currency = "KRW";
      row.quantity = row.quantity ?? 0;
    } catch (err) {
      nameEl.textContent = "조회실패";
      priceEl.textContent = "-";
      amountEl.textContent = "-";
      failedTickers.push(`${row.ticker}(${err.message})`);
      
      // 실패 시 데이터 객체 업데이트
      row.name = "조회실패";
      row.price = null;
      row.amount = null;
      row.quantity = row.quantity ?? 0;
    }
  }

  // 데이터 정렬: 1순위 종목명(오름차순), 2순위 보유금액(내림차순)
  data.sort((a, b) => {
    const nameA = a.name || "";
    const nameB = b.name || "";
    const nameSort = nameA.localeCompare(nameB);
    
    if (nameSort !== 0) {
      return nameSort;
    }
    
    const amountA = a.amount || 0;
    const amountB = b.amount || 0;
    return amountB - amountA;
  });

  // 정렬된 순서대로 테이블 재구성
  tbody.innerHTML = "";
  data.forEach((row) => {
    tbody.appendChild(createRow(row));
  });

  totalEl.textContent = formatMoney(total, "₩");
  if (failedTickers.length > 0) {
    errorEl.textContent = `조회 실패 종목: ${failedTickers.join(", ")}`;
  }
  if (data.length === 0) tbody.appendChild(createRow());
  updateSubtotals(data);
  setLoading(false);
  syncSideCardHeight();
}

async function handleSaveAll() {
  errorEl.textContent = "";
  setLoading(true, "저장 및 재계산 중...");
  const rows = Array.from(tbody.querySelectorAll("tr"));
  const items = [];
  for (const row of rows) {
    const assetType = row
      .querySelector("td:nth-child(1) input")
      .value.trim();
    const ticker = row
      .querySelector("td:nth-child(2) input")
      .value.trim();
    const quantity = Number(
      row.querySelector("td:nth-child(5) input").value || 0,
    );
    if (!ticker) {
      continue;
    }
    items.push({ asset_type: assetType || "주식", ticker, quantity });
  }

  const res = await fetch("/api/holdings/bulk_replace", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    errorEl.textContent = data.detail || "저장 실패";
    setLoading(false);
    return;
  }
  await loadHoldings();
}

function recalcTotal() {
  const rows = Array.from(tbody.querySelectorAll("tr"));
  let total = 0;
  rows.forEach((row) => {
    const amountEl = row.querySelector(".amount");
    const value = parseAmount(amountEl.textContent);
    if (value !== null) total += value;
  });
  totalEl.textContent = formatMoney(total, "₩");
  updateSubtotalsFromDom();
}

async function handleDelete(row) {
  const id = row.dataset.id;
  row.remove();
  recalcTotal();
  if (!id) return;
  const res = await fetch(`/api/holdings/${id}`, { method: "DELETE" });
  if (!res.ok) {
    errorEl.textContent = "삭제 실패";
  }
}

document
  .getElementById("saveAll")
  .addEventListener("click", handleSaveAll);
  
document
  .getElementById("refresh")
  .addEventListener("click", loadHoldings);

let toastTimer = null;
function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.add("show");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toastEl.classList.remove("show");
  }, 2200);
}

document
  .getElementById("importCsvText")
  .addEventListener("click", async () => {
    if (!currentUser) {
      window.location.href = "/auth/login";
      return;
    }
    errorEl.textContent = "";
    const csvTextEl = document.getElementById("csvText");
    const replaceAllEl = document.getElementById("replaceAll");
    let text = csvTextEl.value.trim();
    let replace = replaceAllEl.checked;
    if (!text) {
      const holdingsRes = await fetch("/api/holdings_raw");
      if (!holdingsRes.ok) {
        errorEl.textContent = "보유 자산 상태를 확인하지 못했습니다.";
        return;
      }
      const holdings = await holdingsRes.json();
      if (holdings.length !== 0) {
        errorEl.textContent = "CSV 텍스트를 붙여넣으세요.";
        return;
      }

      const confirmed = window.confirm(
        "입력값이 없습니다. 샘플값으로 수행하시겠습니까?",
      );
      if (!confirmed) {
        return;
      }
      text = SAMPLE_CSV_TEXT;
      replace = true;
      csvTextEl.value = SAMPLE_CSV_TEXT;
      replaceAllEl.checked = true;
    }

    setLoading(true, "CSV 가져오는 중...");
    if (!text) {
      setLoading(false);
      return;
    }
    const res = await fetch(`/api/import_csv_text?replace=${replace}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ csv: text }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      errorEl.textContent = data.detail || "CSV 붙여넣기 실패";
      setLoading(false);
      return;
    }
    csvTextEl.value = "";
    await loadHoldings();
  });

async function fetchAuth() {
  const res = await fetch("/auth/me");
  if (!res.ok) return null;
  return res.json();
}

function setAuthUI(user) {
  currentUser = user;
  authActionsEl.innerHTML = "";
  const adminMenuEl = document.getElementById("adminMenu");
  
  if (!user) {
    authUserEl.textContent = LOGIN_REQUIRED_MESSAGE;
    const loginBtn = document.createElement("button");
    loginBtn.textContent = "Google로 로그인";
    loginBtn.addEventListener("click", () => {
      window.location.href = "/auth/login";
    });
    authActionsEl.appendChild(loginBtn);
    saveAllBtn.disabled = true;
    refreshBtn.disabled = true;
    importBtn.disabled = false;
    // adminMenuEl.classList.remove("show"); // Don't hide the container
    return;
  }

  authUserEl.innerHTML = "";
  if (user.picture) {
    const img = document.createElement("img");
    img.src = user.picture;
    img.alt = "프로필";
    authUserEl.appendChild(img);
  }
  const span = document.createElement("span");
  span.textContent = (user.name || user.email) + "님으로 로그인됨";
  authUserEl.appendChild(span);

  const logoutBtn = document.createElement("button");
  logoutBtn.textContent = "로그아웃";
  logoutBtn.className = "ghost";
  logoutBtn.addEventListener("click", async () => {
    await fetch("/auth/logout", { method: "POST" });
    window.location.reload();
  });
  authActionsEl.appendChild(logoutBtn);
  saveAllBtn.disabled = false;
  refreshBtn.disabled = false;
  importBtn.disabled = false;

  // 관리자 메뉴 표시 (is_admin 필드가 있고 true일 경우)
  if (user.is_admin) {
    adminMenuEl.classList.add("show");
  } else {
    adminMenuEl.classList.remove("show");
  }
}

// Admin Dashboard Functions
let currentEditingUserId = null;

function showAdminDashboard() {
  document.getElementById("adminDashboard").classList.add("show");
  loadUsers();
}

function hideAdminDashboard() {
  document.getElementById("adminDashboard").classList.remove("show");
}

async function loadUsers() {
  try {
    const res = await fetch("/admin/users");
    if (!res.ok) {
      if (res.status === 403) {
        showToast("관리자 권한이 필요합니다.");
        hideAdminDashboard();
        return;
      }
      throw new Error("사용자 목록을 불러오지 못했습니다.");
    }
    const users = await res.json();
    renderUserTable(users);
  } catch (error) {
    console.error("Error loading users:", error);
    showToast(error.message);
  }
}

function renderUserTable(users) {
  const tbody = document.getElementById("userTableBody");
  tbody.innerHTML = "";

  if (users.length === 0) {
    tbody.innerHTML = "<tr><td colspan='5' style='text-align: center; padding: 20px;'>사용자가 없습니다.</td></tr>";
    return;
  }

  users.forEach(user => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>
        <div class="user-info">
          ${user.picture ? `<img src="${user.picture}" alt="프로필" class="user-avatar">` : ''}
          <span>${user.name || '이름 없음'}</span>
          ${user.is_admin ? '<span class="admin-badge">관리자</span>' : ''}
        </div>
      </td>
      <td>${user.email}</td>
      <td>${new Date(user.created_at).toLocaleDateString('ko-KR')}</td>
      <td>${user.is_admin ? '예' : '아니오'}</td>
      <td>
        <div class="user-actions">
          <button class="btn-small btn-edit" onclick="editUser(${user.id})">수정</button>
          <button class="btn-small btn-delete" onclick="deleteUser(${user.id})">삭제</button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function showUserModal(title = "사용자 추가", user = null) {
  currentEditingUserId = user ? user.id : null;
  document.getElementById("modalTitle").textContent = title;
  document.getElementById("userEmail").value = user ? user.email : "";
  document.getElementById("userName").value = user ? user.name || "" : "";
  document.getElementById("userIsAdmin").checked = user ? user.is_admin : false;
  document.getElementById("userModal").classList.add("show");
}

function hideUserModal() {
  document.getElementById("userModal").classList.remove("show");
  document.getElementById("userForm").reset();
  currentEditingUserId = null;
}

async function saveUser(formData) {
  try {
    const url = currentEditingUserId ? `/admin/users/${currentEditingUserId}` : "/admin/users";
    const method = currentEditingUserId ? "PATCH" : "POST";
    
    const res = await fetch(url, {
      method: method,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formData)
    });
    
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(error.detail || "사용자 저장에 실패했습니다.");
    }

    showToast(currentEditingUserId ? "사용자 정보가 수정되었습니다." : "사용자가 추가되었습니다.");
    hideUserModal();
    loadUsers();
  } catch (error) {
    console.error("Error saving user:", error);
    showToast(error.message);
  }
}

async function deleteUser(userId) {
  if (!confirm("정말로 이 사용자를 삭제하시겠습니까? 관련된 모든 데이터가 삭제됩니다.")) {
    return;
  }

  try {
    const res = await fetch(`/admin/users/${userId}`, { method: "DELETE" });
    if (!res.ok) {
      const error = await res.json().catch(() => ({}));
      throw new Error(error.detail || "사용자 삭제에 실패했습니다.");
    }
    showToast("사용자가 삭제되었습니다.");
    loadUsers();
  } catch (error) {
    console.error("Error deleting user:", error);
    showToast(error.message);
  }
}

function editUser(userId) {
  // 현재 테이블에서 사용자 정보 찾기
  const rows = document.querySelectorAll("#userTableBody tr");
  for (let row of rows) {
    const editBtn = row.querySelector(".btn-edit");
    if (editBtn && editBtn.onclick.toString().includes(userId)) {
      const cells = row.querySelectorAll("td");
      const userInfo = cells[0].querySelector("span").textContent;
      const email = cells[1].textContent;
      const isAdmin = cells[3].textContent === "예";
      
      showUserModal("사용자 수정", {
        id: userId,
        name: userInfo !== "이름 없음" ? userInfo : "",
        email: email,
        is_admin: isAdmin
      });
      break;
    }
  }
}

function showToast(message) {
  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;
  document.body.appendChild(toast);
  
  setTimeout(() => toast.classList.add("show"), 10);
  setTimeout(() => {
    toast.classList.remove("show");
    setTimeout(() => document.body.removeChild(toast), 200);
  }, 3000);
}

// 직접 저장 핸들러
function handleSaveUser(event) {
  event.preventDefault();
  const formData = {
    email: document.getElementById("userEmail").value,
    name: document.getElementById("userName").value,
    is_admin: document.getElementById("userIsAdmin").checked
  };
  saveUser(formData);
}

// Event Listeners for Admin Dashboard
const adminBtn = document.getElementById("adminBtn");
if (adminBtn) adminBtn.addEventListener("click", showAdminDashboard);

const closeAdminBtn = document.getElementById("closeAdminBtn");
if (closeAdminBtn) closeAdminBtn.addEventListener("click", hideAdminDashboard);

const addUserBtn = document.getElementById("addUserBtn");
if (addUserBtn) addUserBtn.addEventListener("click", () => showUserModal());

const refreshUsersBtn = document.getElementById("refreshUsersBtn");
if (refreshUsersBtn) refreshUsersBtn.addEventListener("click", loadUsers);

// 모달 닫기 버튼 - 이벤트 위임 방식 사용
document.addEventListener("click", (e) => {
  if (e.target.id === "closeModalBtn" || e.target.id === "cancelModalBtn") {
    e.preventDefault();
    hideUserModal();
  }
  
  // 저장 버튼도 이벤트 위임으로 처리
  if (e.target.type === "submit" && e.target.form && e.target.form.id === "userForm") {
    e.preventDefault();
    const formData = {
      email: document.getElementById("userEmail").value,
      name: document.getElementById("userName").value,
      is_admin: document.getElementById("userIsAdmin").checked
    };
    saveUser(formData);
  }
});

const userForm = document.getElementById("userForm");
if (userForm) {
  userForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = {
      email: document.getElementById("userEmail").value,
      name: document.getElementById("userName").value,
      is_admin: document.getElementById("userIsAdmin").checked
    };
    await saveUser(formData);
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  const isLocalhost =
    window.location.hostname === '127.0.0.1' ||
    window.location.hostname === 'localhost';

  // PWA Service Worker Registration
  if ('serviceWorker' in navigator) {
    if (isLocalhost) {
      navigator.serviceWorker.getRegistrations()
        .then((registrations) => Promise.all(registrations.map((registration) => registration.unregister())))
        .catch((err) => console.log('SW cleanup failed', err));
    } else {
      window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
          .then(reg => console.log('SW registered', reg))
          .catch(err => console.log('SW registration failed', err));
      });
    }
  }

  // PWA Installation Logic
  let deferredPrompt;
  const pwaInstallBtn = document.getElementById('pwaInstallBtn');

  if (isLocalhost) {
    document.querySelector('link[rel="manifest"]')?.remove();
    if (pwaInstallBtn) {
      pwaInstallBtn.style.display = 'none';
    }
  }

  // iOS Safari detection
  const isIos = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
  const isStandalone = window.matchMedia('(display-mode: standalone)').matches;

  if (!isLocalhost && isIos && !isStandalone) {
    // iOS Safari typically needs manual instructions
    if (pwaInstallBtn) {
      pwaInstallBtn.textContent = '앱 설치 방법(iOS)';
      pwaInstallBtn.style.display = 'inline-flex';
      pwaInstallBtn.addEventListener('click', () => {
        alert('아이폰(iOS)에서 설치하려면:\n1. 브라우저 하단의 [공유] 버튼을 누르세요.\n2. [홈 화면에 추가]를 선택하세요.');
      });
    }
  }

  window.addEventListener('beforeinstallprompt', (e) => {
    if (isLocalhost) return;
    e.preventDefault();
    deferredPrompt = e;
    if (pwaInstallBtn) {
      pwaInstallBtn.textContent = '앱 설치하기';
      pwaInstallBtn.style.display = 'inline-flex';
    }
  });

  if (pwaInstallBtn) {
    pwaInstallBtn.addEventListener('click', async () => {
      if (!deferredPrompt) return;
      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      deferredPrompt = null;
      pwaInstallBtn.style.display = 'none';
    });
  }

  window.addEventListener('appinstalled', () => {
    if (pwaInstallBtn) {
      pwaInstallBtn.style.display = 'none';
    }
  });

  window.addEventListener("resize", syncSideCardHeight);
  syncSideCardHeight();

  try {
    const user = await fetchAuth();
    setAuthUI(user);
    if (user) {
      await loadHoldings();
    }
    syncSideCardHeight();
  } catch (error) {
    console.error('Auth initialization error:', error);
    // Fallback: show login button manually
    const authActionsEl = document.getElementById("authActions");
    if (authActionsEl) {
      const loginBtn = document.createElement("button");
      loginBtn.textContent = "Google로 로그인";
      loginBtn.addEventListener("click", () => {
        window.location.href = "/auth/login";
      });
      authActionsEl.appendChild(loginBtn);
    }
    syncSideCardHeight();
  }
});
