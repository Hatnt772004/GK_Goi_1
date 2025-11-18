# ================================================================
# Thống kê mô tả & EDA hoàn chỉnh (SỬ DỤNG ADJUSTED CLOSE & VẼ BIỂU ĐỒ)
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
import scipy.stats as stats 
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch

# Tắt các cảnh báo
warnings.filterwarnings('ignore')

print("=== Đang tải và chuẩn bị dữ liệu ===")

# --- 1. CẤU HÌNH & ĐỌC DỮ LIỆU ---
DATA_FOLDER = 'data_ohlcv'
FILE_NAME = 'data_final.csv'
FILE_PATH = os.path.join(DATA_FOLDER, FILE_NAME)

# Cấu hình thư mục lưu ảnh
PLOT_FOLDER = os.path.join(DATA_FOLDER, 'plots')
os.makedirs(PLOT_FOLDER, exist_ok=True)
print(f"📂 Các biểu đồ sẽ được lưu vào: {PLOT_FOLDER}")

try:
    if not os.path.exists(FILE_PATH):
        print(f"❌ Lỗi: Không tìm thấy tệp '{FILE_PATH}'.")
        sys.exit()
        
    df_full = pd.read_csv(FILE_PATH)
    print(f"✅ Đã tải thành công file: {FILE_PATH}")
except Exception as e:
    print(f"❌ Lỗi khi đọc file: {e}")
    sys.exit()

# --- 2. Chuẩn hóa tên cột ---
df_full.columns = [c.strip().lower() for c in df_full.columns]

# --- 3. Kiểm tra và chuyển cột ngày ---
if 'date' in df_full.columns:
    df_full['date'] = pd.to_datetime(df_full['date'])
elif 'Date' in df_full.columns:
    df_full['date'] = pd.to_datetime(df_full['Date'])
else:
    df_full.iloc[:, 0] = pd.to_datetime(df_full.iloc[:, 0])
    df_full.rename(columns={df_full.columns[0]: 'date'}, inplace=True)

df_full = df_full.sort_values('date').set_index('date')

# --- 4. Chuyển đổi cấu trúc (Long -> Wide) ---
# QUAN TRỌNG: Sử dụng 'adjusted_close' thay vì 'close'
if 'ticker' in df_full.columns:
    if 'adjusted_close' in df_full.columns:
        value_col = 'adjusted_close'
        print("✅ Đang sử dụng cột 'adjusted_close' để tính toán (Chính xác).")
    elif 'close' in df_full.columns:
        value_col = 'close'
        print("⚠️ Cảnh báo: Không thấy 'adjusted_close', đang dùng 'close'.")
    else:
        print("❌ Lỗi: Không tìm thấy cột giá.")
        sys.exit()

    df_temp = df_full.reset_index() 
    price_wide = df_temp.pivot(index='date', columns='ticker', values=value_col).sort_index()
else:
    print("❌ Lỗi cấu trúc dữ liệu: Thiếu cột 'ticker'.")
    sys.exit()

# --- 5. Chọn nhóm ngân hàng ---
bank_tickers = ['ACB','BID','CTG','EIB','HDB','LPB','MBB','SHB',
                'STB','TCB','TPB','VCB','VIB','VPB']
available_banks = [c for c in bank_tickers if c in price_wide.columns]

if not available_banks:
    print("⚠️ Cảnh báo: Không tìm thấy mã ngân hàng nào.")
    available_banks = list(price_wide.columns)[:14]

prices = price_wide[available_banks].sort_index().copy()
print(f"✅ Số mã ngân hàng sử dụng: {len(available_banks)}")

# --- 6. Thống kê missing ---
summary_dates = pd.DataFrame({
    'first_valid': prices.apply(lambda x: x.first_valid_index()),
    'last_valid': prices.apply(lambda x: x.last_valid_index()),
    'missing_count': prices.isna().sum()
})

# --- 7. Tính returns (Dựa trên Adjusted Close) ---
returns = prices.pct_change().dropna(how='all')
log_returns = np.log(prices).diff().dropna(how='all')

# --- 8. Thống kê mô tả ---
price_stats = prices.describe().T
return_stats = returns.describe().T
corr_returns = returns.corr()

# ============================================================
# 📊 BẢNG THỐNG KÊ CẤU TRÚC LỢI SUẤT — BANKS & VNI
# ============================================================

# 1️⃣ Xác định cột VNI (VN-Index)
if 'vni_log_return' in df_full.columns:
    vni_returns_series = df_full['vni_log_return'].groupby('date').first().dropna()
    print("✅ Đã lấy dữ liệu VNINDEX từ cột vni_log_return.")
else:
    vni_returns_series = None
    print("⚠️ Không tìm thấy dữ liệu VNINDEX.")

# 2️⃣ Tính các đặc trưng thống kê
return_summary = pd.DataFrame({
    'Mean': log_returns.mean(),
    'Std': log_returns.std(),
    'Skewness': log_returns.skew(),
    'Kurtosis': log_returns.kurt()
})

# 3️⃣ Tính thêm hàng VNI (nếu có)
if vni_returns_series is not None:
    vni_row = pd.DataFrame({
        'Mean': [vni_returns_series.mean()],
        'Std': [vni_returns_series.std()],
        'Skewness': [vni_returns_series.skew()],
        'Kurtosis': [vni_returns_series.kurt()]
    }, index=['VNINDEX'])
    return_summary = pd.concat([return_summary, vni_row])

return_summary = return_summary.round(6)
print("\n📋 BẢNG THỐNG KÊ CẤU TRÚC LỢI SUẤT:")
print(return_summary)

# Lưu bảng
save_path_struct = os.path.join(DATA_FOLDER, "log_return_structure_summary.csv")
return_summary.to_csv(save_path_struct)

# --- 9. Equal-weighted index ---
ew_bank = prices.mean(axis=1).rename('EW_BANKS')
cum_returns = (1 + returns).cumprod() - 1
rolling_vol_21 = returns.rolling(21).std() * np.sqrt(252)
rolling_vol_ew_21 = rolling_vol_21.mean(axis=1)

# --- 10. Drawdown EW_BANKS ---
wealth = (1 + ew_bank.pct_change().fillna(0)).cumprod()
drawdown = (wealth - wealth.cummax()) / wealth.cummax()

# ============================================================
# 💾 Lưu kết quả thống kê ra CSV
# ============================================================
price_stats.to_csv(os.path.join(DATA_FOLDER, "price_summary.csv"))
return_stats.to_csv(os.path.join(DATA_FOLDER, "return_summary.csv"))
corr_returns.to_csv(os.path.join(DATA_FOLDER, "return_correlation.csv"))

# ============================================================
# 🎨 VẼ BIỂU ĐỒ VÀ LƯU (PHẦN BẠN ĐANG THIẾU)
# ============================================================
print("\n🎨 Đang vẽ và lưu biểu đồ (Adjusted Close)...")
plt.style.use('seaborn-v0_8-whitegrid') # Nền trắng

def save_plot(filename):
    save_path = os.path.join(PLOT_FOLDER, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Đã lưu: {filename}")
    plt.close()

# 1. Adjusted Price Series
plt.figure(figsize=(12,6))
for c in prices.columns:
    plt.plot(prices.index, prices[c], label=c)
plt.title("Adjusted Price Series — Bank Stocks")
plt.xlabel("Date"); plt.ylabel("Adjusted Price"); plt.legend(ncol=4, fontsize='small')
plt.tight_layout()
save_plot("01_Adjusted_Price_Series.png")

# 2. Normalized Prices
normalized = prices.divide(prices.ffill().iloc[0]).mul(100)
plt.figure(figsize=(12,6))
for c in normalized.columns:
    plt.plot(normalized.index, normalized[c], label=c)
plt.title("Normalized Prices (Base 100)")
plt.xlabel("Date"); plt.ylabel("Index (Base 100)")
plt.legend(ncol=4, fontsize='small'); plt.tight_layout()
save_plot("02_Normalized_Prices.png")

# 3. Cumulative Returns
plt.figure(figsize=(12,6))
for c in cum_returns.columns:
    plt.plot(cum_returns.index, cum_returns[c], label=c)
plt.title("Cumulative Returns (Based on Adj Close)")
plt.xlabel("Date"); plt.ylabel("Cumulative Return")
plt.legend(ncol=4, fontsize='small'); plt.tight_layout()
save_plot("03_Cumulative_Returns.png")

# 4. Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_returns, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
plt.title("Return Correlation Matrix", fontsize=13, weight='bold')
plt.tight_layout()
save_plot("04_Correlation_Matrix.png")

# 5. Boxplot
plt.figure(figsize=(12,6))
sns.boxplot(data=returns, orient='v')
plt.title("Boxplot of Daily Returns", fontsize=13, weight='bold')
plt.ylabel("Daily Return"); plt.xticks(rotation=90)
plt.tight_layout()
save_plot("05_Boxplot_Returns.png")

# 6. Rolling Volatility
plt.figure(figsize=(12,5))
plt.plot(rolling_vol_ew_21.index, rolling_vol_ew_21, color='blue', linewidth=1.5)
plt.title("Average Rolling Annualized Volatility (21-day)", fontsize=13, weight='bold')
plt.xlabel("Date"); plt.ylabel("Annualized Volatility")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
save_plot("06_Rolling_Volatility.png")

# 7. Distribution (KDE)
plt.figure(figsize=(12, 6))
for c in log_returns.columns:
    sns.kdeplot(log_returns[c].dropna(), label=c, fill=True, alpha=0.3, linewidth=1.2)
plt.title("KDE Plot of Log Returns", fontsize=13, weight='bold')
plt.xlabel("Log Return"); plt.ylabel("Density"); plt.legend(ncol=3, fontsize='small')
plt.tight_layout()
save_plot("07_Log_Returns_Distribution.png")

# 8. Histogram Fat Tails
ew_log_return = log_returns[available_banks].mean(axis=1).dropna()
plt.figure(figsize=(10, 5))
sns.histplot(ew_log_return, bins=50, kde=True, color='skyblue', stat='density', edgecolor='white', alpha=0.5, label='Empirical')
x = np.linspace(ew_log_return.min(), ew_log_return.max(), 300)
kde = stats.gaussian_kde(ew_log_return)
y = kde(x)
plt.fill_between(x, y, color='skyblue', alpha=0.4, label='KDE Density Band')
mu, sigma = ew_log_return.mean(), ew_log_return.std()
x = np.linspace(ew_log_return.min(), ew_log_return.max(), 200)
plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r--', label='Normal PDF')
plt.title("Histogram of EW_BANKS Log Returns — Fat Tails Test", fontsize=13, weight='bold')
plt.xlabel("Daily Log Return"); plt.ylabel("Density"); plt.legend()
plt.tight_layout()
save_plot("08_Histogram_Fat_Tails.png")

# 9. Drawdown
plt.figure(figsize=(12,5))
plt.plot(drawdown.index, drawdown, color='red')
plt.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
plt.title("Drawdown — Equal-Weighted Bank Index", fontsize=13, weight='bold')
plt.xlabel("Date"); plt.ylabel("Drawdown")
plt.tight_layout()
save_plot("09_Drawdown.png")

# ============================================================
# KIỂM ĐỊNH THỐNG KÊ
# ============================================================

# ADF Test
adf_results = {}
for c in log_returns.columns:
    series = log_returns[c].dropna()
    try:
        adf_stat, p_value, _, _, _, _ = adfuller(series)
        adf_results[c] = {"ADF Statistic": adf_stat, "p-value": p_value}
    except: pass
adf_summary = pd.DataFrame(adf_results).T
adf_summary["Stationary"] = adf_summary["p-value"] < 0.05
adf_summary.to_csv(os.path.join(DATA_FOLDER, "ADF_stationarity_summary.csv"))

# ARCH Test
arch_results = {}
for c in log_returns.columns:
    series = log_returns[c].dropna()
    try:
        lm_stat, lm_pvalue, _, _ = het_arch(series)
        arch_results[c] = {"LM Stat": lm_stat, "p-value": lm_pvalue}
    except: pass
arch_summary = pd.DataFrame(arch_results).T
arch_summary["ARCH Effect Present"] = arch_summary["p-value"] < 0.05
arch_summary.to_csv(os.path.join(DATA_FOLDER, "ARCH_effect_summary.csv"))

print("\n" + "="*60)
print(f"✅ HOÀN TẤT! Kiểm tra thư mục: {PLOT_FOLDER}")
print("="*60)