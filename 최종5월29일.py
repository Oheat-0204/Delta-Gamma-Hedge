#%%
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, jarque_bera, chi2
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
import koreanize_matplotlib

years = 3
ticker = 'SPY'
end_date = datetime.today()
start_date = end_date - timedelta(days=years*365)
data = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval='1d')

# 로그수익률
data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
log_returns = data['LogReturn'].dropna()

jb_stat, jb_pvalue = jarque_bera(log_returns)
print(f"Jarque-Bera test statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}")
#정규성 충족 X

#%%

# 점프 시기 탐지 
lower_q, upper_q = log_returns.quantile([0.025, 0.975])
jump_mask = (log_returns < lower_q) | (log_returns > upper_q)
jump_times = log_returns[jump_mask].index
jump_sizes = log_returns[jump_mask]

# 포아송 분포 람다 추정 (연간 점프강도) 및 점프 크기 평균(κ), 표준편차(δ) 추정
n_obs = len(log_returns)
jump_count = len(jump_sizes)
lambda_hat = jump_count / (n_obs / 252)   # 1년 환산
kappa = jump_sizes.mean()
delta = jump_sizes.std(ddof=1)

T = 30 / 252        
r = 0.04               
vol = log_returns.std() * np.sqrt(252)

print(lambda_hat)
def bs_put_price(S,K,T,r,sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def bs_put_delta(S,K,T,r,sigma):
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    return norm.cdf(d1) - 1

def merton_jump_put_price(S,K,T,r,sigma,lam,kappa,delta,N=50):
    price = 0.0
    for n in range(N+1):
        r_n     = r - lam*kappa + (n*np.log(1+kappa))/T
        sigma_n = np.sqrt(sigma**2 + n*delta**2/T)
        w       = np.exp(-lam*T)*(lam*T)**n/math.factorial(n)
        price  += w * bs_put_price(S,K,T,r_n,sigma_n)
    return price

def merton_jump_put_delta(S,K,T,r,sigma,lam,kappa,delta,N=50):
    d_mjd = 0.0
    for n in range(N+1):
        r_n     = r - lam*kappa + (n*np.log(1+kappa))/T
        sigma_n = np.sqrt(sigma**2 + n*delta**2/T)
        w       = np.exp(-lam*T)*(lam*T)**n/math.factorial(n)
        d_mjd  += w * bs_put_delta(S,K,T,r_n,sigma_n)
    return d_mjd

def merton_jump_put_gamma(S,K,T,r,sigma,lam,kappa,delta,N=50):
    g_mjd = 0.0
    for n in range(N+1):
        r_n     = r - lam*kappa + (n*np.log(1+kappa))/T
        sigma_n = np.sqrt(sigma**2 + n*delta**2/T)
        w       = np.exp(-lam*T)*(lam*T)**n/math.factorial(n)
        # BS Gamma
        d1 = (np.log(S/K)+(r_n+0.5*sigma_n**2)*T)/(sigma_n*np.sqrt(T))
        g_bs= norm.pdf(d1)/(S*sigma_n*np.sqrt(T))
        g_mjd+= w*g_bs
    return g_mjd

#%% 정적 전략별 P/L 계산
option_multiplier = 100
T_days = 30
T = T_days / 252

# 결과 저장
dynamic_hedge_records = {}


jump_times_filtered = jump_times[jump_times >= data.index[0]]

for jump_date in jump_times_filtered:
    i0 = data.index.get_loc(jump_date)-1
    if i0 + T_days >= len(data):  # 데이터 부족 시 제외
        continue

    date_window = data.index[i0: i0 + T_days + 1]
    S_window = data['Close'].iloc[i0: i0 + T_days + 1].values

    K = S_window[0] * 0.936
    lam_t = lambda_hat

    daily_pls = []
    delta_prev = None
    cashflow = 0
    option_cost = merton_jump_put_price(S_window[0], K, T, r, vol, lam_t, kappa, delta) * option_multiplier

    for t in range(T_days):
        S_t = S_window[t]
        S_next = S_window[t+1]
        T_remain = (T_days - t) / 252

        opt_t = merton_jump_put_price(S_t, K, T_remain, r, vol, lam_t, kappa, delta) * option_multiplier
        opt_next = merton_jump_put_price(S_next, K, T_remain - 1/252, r, vol, lam_t, kappa, delta) * option_multiplier
        delta_t = merton_jump_put_delta(S_t, K, T_remain, r, vol, lam_t, kappa, delta)

        hedge_units = -delta_t * option_multiplier

        if t == 0:
            hedge_cost = hedge_units * S_t
            cashflow += hedge_cost
            delta_prev = hedge_units
            rebal_cost = 0
        else:
            rebal_units = hedge_units - delta_prev
            rebal_cost = rebal_units * S_t
            cashflow += rebal_cost
            delta_prev = hedge_units

        stock_pl = delta_prev * (S_next - S_t)
        option_pl = opt_next - opt_t
        net_pl = stock_pl + option_pl - rebal_cost

        daily_pls.append(net_pl)

    dynamic_hedge_records[jump_date] = daily_pls



#%%
# jump_date (key) 별로 일별 P/L 리스트 (value)
print(dynamic_hedge_records.keys())  # 점프일자들

for date, pl_list in dynamic_hedge_records.items():
    cleaned = [pl.item() if isinstance(pl, np.ndarray) else pl for pl in pl_list[:5]]
    print(f"{date.date()} → {cleaned}...")


# %%
# numpy array → float 변환 + 누락된 길이 29 보장
import pandas as pd
import numpy as np

# 누적 결과를 저장할 리스트
records_for_excel = []

# 날짜 정렬
for date in sorted(dynamic_hedge_records.keys()):
    daily_pls = dynamic_hedge_records[date]

    # numpy array → float 변환
    clean_pls = [pl.item() if isinstance(pl, np.ndarray) else pl for pl in daily_pls]

    # 길이 29로 패딩 (마지막 만기일까지 맞추기)
    clean_pls = clean_pls + [np.nan] * (30 - len(clean_pls))

    records_for_excel.append([date.strftime('%Y-%m-%d')] + clean_pls)

# DataFrame 생성
columns = ['Jump Date'] + [f'day {i+1}' for i in range(30)]
df_output = pd.DataFrame(records_for_excel, columns=columns)

# Excel 저장
df_output.to_excel('delta_hedge_daily_PL.xlsx', index=False)

#%% 동적 델타헷지 with 실제 현금 흐름 반영
# 점프일 하나 수동 지정
selected_jump_date = pd.Timestamp('2025-01-15')  # 예시

# 날짜 위치 확인
i0 = data.index.get_loc(selected_jump_date)-1
T_days = 30
if i0 + T_days >= len(data):
    raise ValueError("데이터 부족")

date_window = data.index[i0: i0 + T_days + 1]
S_window = data['Close'].iloc[i0: i0 + T_days + 1].values

K = S_window[0] * 0.936
T = T_days / 252
lam_t = lambda_hat

option_price_0 = merton_jump_put_price(S_window[0], K, T, r, vol, lam_t, kappa, delta)
delta_0 = merton_jump_put_delta(S_window[0], K, T, r, vol, lam_t, kappa, delta)
hedge_units_0 = -delta_0 * option_multiplier
option_cost = option_price_0 * option_multiplier
stock_cost = hedge_units_0 * S_window[0]
initial_cash_outflow = option_cost + stock_cost

# 리밸런싱 누적 현금흐름 포함
cumulative_rebalancing_cashflow = 0
tracking = []
stock_held_prev = hedge_units_0

for t in range(T_days):
    S_t = S_window[t]
    T_remain = (T_days - t) / 252

    delta_t = merton_jump_put_delta(S_t, K, T_remain, r, vol, lam_t, kappa, delta)
    hedge_units = -delta_t * option_multiplier
    Δ_stock_traded = hedge_units - stock_held_prev

    option_price_t = merton_jump_put_price(S_t, K, T_remain, r, vol, lam_t, kappa, delta)
    stock_value = hedge_units * S_t

    
    if t == T_days - 1:
        option_value = max(K - S_t, 0) * option_multiplier
    else:
        option_value = option_price_t * option_multiplier

   
    rebalancing_cf = -Δ_stock_traded * S_t
    cumulative_rebalancing_cashflow += rebalancing_cf

    total_value = stock_value + option_value + cumulative_rebalancing_cashflow
    pnl = total_value - initial_cash_outflow

    tracking.append({
        'Date': date_window[t].date(),
        'Stock Price': float(S_t),
        'Option Delta': float(delta_t),
        'Stock Held': float(hedge_units),
        'Δ Stock Traded': float(Δ_stock_traded),
        'Stock Value': float(stock_value),
        'Option Value': float(option_value),
        'Rebalancing CF': float(rebalancing_cf),
        'Rebalancing CF(누적)': float(cumulative_rebalancing_cashflow),
        'Total Value': float(total_value),
        'PnL': float(pnl)
    })

    stock_held_prev = hedge_units

    

# 엑셀 저장
tracking_df = pd.DataFrame(tracking)
tracking_df.to_excel("delta_hedge.xlsx", index=False)


# 데이터프레임 변환 및 저장
tracking_df = pd.DataFrame(tracking)
tracking_df.to_excel('selected_jump_tracking.xlsx', index=False)

# %%
import pandas as pd

# 초기값 설정 (이전 selected_jump_date 기준)
selected_jump_date = pd.Timestamp('2025-01-15')
i0 = data.index.get_loc(selected_jump_date)-1
T_days = 30
if i0 + T_days >= len(data):
    raise ValueError("데이터 부족")

date_window = data.index[i0: i0 + T_days + 1]
S_window = data['Close'].iloc[i0: i0 + T_days + 1].values

# 행사가
S0 = S_window[0]
K1 = S0 * 0.936  # ATM 풋옵션
K2 = S0 * 0.92   # 8% OTM 풋옵션

T = T_days / 252
lam_t = lambda_hat

# 초기 옵션 가격, 델타, 감마 계산
opt1_price_0 = merton_jump_put_price(S0, K1, T, r, vol, lam_t, kappa, delta)
opt2_price_0 = merton_jump_put_price(S0, K2, T, r, vol, lam_t, kappa, delta)

opt1_delta_0 = merton_jump_put_delta(S0, K1, T, r, vol, lam_t, kappa, delta)
opt2_delta_0 = merton_jump_put_delta(S0, K2, T, r, vol, lam_t, kappa, delta)

opt1_gamma_0 = merton_jump_put_gamma(S0, K1, T, r, vol, lam_t, kappa, delta)
opt2_gamma_0 = merton_jump_put_gamma(S0, K2, T, r, vol, lam_t, kappa, delta)

# 감마 헷지를 위한 옵션 2의 수량 결정 (옵션 1의 감마를 옵션 2로 제거)
n2 = - opt1_gamma_0 / opt2_gamma_0
n1 = 1  # 옵션 1은 1계약
delta_total = n1 * opt1_delta_0 + n2 * opt2_delta_0
stock_units = - delta_total * option_multiplier

# 초기 현금 유출
opt1_cost = n1 * opt1_price_0 * option_multiplier
opt2_cost = n2 * opt2_price_0 * option_multiplier
stock_cost = stock_units * S0
initial_cash_outflow = opt1_cost + opt2_cost + stock_cost

# 시뮬레이션
cumulative_rebalancing_cashflow = 0
tracking_gamma = []
stock_prev = stock_units
n2_prev = n2

for t in range(T_days):
    S_t = S_window[t]
    T_remain = (T_days - t) / 252

    d1 = merton_jump_put_delta(S_t, K1, T_remain, r, vol, lam_t, kappa, delta)
    d2 = merton_jump_put_delta(S_t, K2, T_remain, r, vol, lam_t, kappa, delta)
    g1 = merton_jump_put_gamma(S_t, K1, T_remain, r, vol, lam_t, kappa, delta)
    g2 = merton_jump_put_gamma(S_t, K2, T_remain, r, vol, lam_t, kappa, delta)

    n2 = - g1 / g2
    n1 = 1
    delta_total = n1 * d1 + n2 * d2
    stock_units = - delta_total * option_multiplier

    Δ_stock = stock_units - stock_prev
    rebalancing_cf = -Δ_stock * S_t
    cumulative_rebalancing_cashflow += rebalancing_cf

    opt1_price_t = merton_jump_put_price(S_t, K1, T_remain, r, vol, lam_t, kappa, delta)
    opt2_price_t = merton_jump_put_price(S_t, K2, T_remain, r, vol, lam_t, kappa, delta)

    if t == T_days - 1:
        opt1_value = max(K1 - S_t, 0) * option_multiplier
        opt2_value = max(K2 - S_t, 0) * option_multiplier * n2
    else:
        opt1_value = opt1_price_t * option_multiplier
        opt2_value = opt2_price_t * option_multiplier * n2

    stock_value = stock_units * S_t
    total_value = stock_value + opt1_value + opt2_value + cumulative_rebalancing_cashflow
    pnl = total_value - initial_cash_outflow

    tracking_gamma.append({
        'Date': date_window[t].date(),
        'Stock Price': float(S_t),
        'Option1 Delta': float(d1),
        'Option2 Delta': float(d2),
        'Option1 Gamma': float(g1),
        'Option2 Gamma': float(g2),
        'Option2 Held': float(n2),
        'Stock Held': float(stock_units),
        'Δ Stock Traded': float(Δ_stock),
        'Option1 Value': float(opt1_value),
        'Option2 Value': float(opt2_value),
        'Stock Value': float(stock_value),
        'Rebalancing CF': float(rebalancing_cf),
        'Rebalancing CF(누적)': float(cumulative_rebalancing_cashflow),
        'Total Value': float(total_value),
        'PnL': float(pnl)
    })

    stock_prev = stock_units
    n2_prev = n2

# 저장
df_gamma = pd.DataFrame(tracking_gamma)
df_gamma.to_excel("gamma_hedge.xlsx", index=False)

# %%
import pandas as pd

option_multiplier = 100
T_days = 30
T = T_days / 252

summary_records = []

for jump_date in jump_times_filtered:
    i0 = data.index.get_loc(jump_date)-1
    if i0 + T_days >= len(data):
        continue

    date_window = data.index[i0: i0 + T_days + 1]
    S_window = data['Close'].iloc[i0: i0 + T_days + 1].values

    S0 = S_window[0]
    K1 = S0 * 0.936
    K2 = S0 * 0.92

    # 델타 헷지 초기 설정
    opt_price_0 = merton_jump_put_price(S0, K1, T, r, vol, lambda_hat, kappa, delta)
    delta_0 = merton_jump_put_delta(S0, K1, T, r, vol, lambda_hat, kappa, delta)
    stock_units_delta = -delta_0 * option_multiplier
    initial_cf_delta = opt_price_0 * option_multiplier + stock_units_delta * S0
    cumulative_cf_delta = 0
    delta_prev = stock_units_delta

    for t in range(T_days):
        S_t = S_window[t]
        T_remain = (T_days - t) / 252
        delta_t = merton_jump_put_delta(S_t, K1, T_remain, r, vol, lambda_hat, kappa, delta)
        hedge_units = -delta_t * option_multiplier
        Δ_stock = hedge_units - delta_prev
        rebal_cf = -Δ_stock * S_t
        cumulative_cf_delta += rebal_cf
        delta_prev = hedge_units

    S_T = S_window[-1]
    opt_value_delta = max(K1 - S_T, 0) * option_multiplier
    stock_value_delta = hedge_units * S_T
    total_value_delta = stock_value_delta + opt_value_delta + cumulative_cf_delta
    pnl_delta = total_value_delta - initial_cf_delta

    # 감마 헷지 초기 설정
    d1 = merton_jump_put_delta(S0, K1, T, r, vol, lambda_hat, kappa, delta)
    d2 = merton_jump_put_delta(S0, K2, T, r, vol, lambda_hat, kappa, delta)
    g1 = merton_jump_put_gamma(S0, K1, T, r, vol, lambda_hat, kappa, delta)
    g2 = merton_jump_put_gamma(S0, K2, T, r, vol, lambda_hat, kappa, delta)
    n2 = -g1 / g2
    delta_total = d1 + n2 * d2
    stock_units_gamma = -delta_total * option_multiplier

    opt1_price = merton_jump_put_price(S0, K1, T, r, vol, lambda_hat, kappa, delta)
    opt2_price = merton_jump_put_price(S0, K2, T, r, vol, lambda_hat, kappa, delta)
    opt1_cost = opt1_price * option_multiplier
    opt2_cost = opt2_price * option_multiplier * n2
    stock_cost = stock_units_gamma * S0
    initial_cf_gamma = opt1_cost + opt2_cost + stock_cost

    cumulative_cf_gamma = 0
    stock_prev = stock_units_gamma

    for t in range(T_days):
        S_t = S_window[t]
        T_remain = (T_days - t) / 252
        d1 = merton_jump_put_delta(S_t, K1, T_remain, r, vol, lambda_hat, kappa, delta)
        d2 = merton_jump_put_delta(S_t, K2, T_remain, r, vol, lambda_hat, kappa, delta)
        g1 = merton_jump_put_gamma(S_t, K1, T_remain, r, vol, lambda_hat, kappa, delta)
        g2 = merton_jump_put_gamma(S_t, K2, T_remain, r, vol, lambda_hat, kappa, delta)
        n2 = -g1 / g2
        delta_total = d1 + n2 * d2
        stock_units = -delta_total * option_multiplier
        Δ_stock = stock_units - stock_prev
        rebal_cf = -Δ_stock * S_t
        cumulative_cf_gamma += rebal_cf
        stock_prev = stock_units

    opt1_val = max(K1 - S_T, 0) * option_multiplier
    opt2_val = max(K2 - S_T, 0) * option_multiplier * n2
    stock_val = stock_units * S_T
    total_val_gamma = stock_val + opt1_val + opt2_val + cumulative_cf_gamma
    pnl_gamma = total_val_gamma - initial_cf_gamma

    summary_records.append({
    'Jump Date': jump_date.strftime('%Y-%m-%d'),
    'Delta Hedge PnL': float(pnl_delta),
    'Gamma Hedge PnL': float(pnl_gamma)


    })

summary_df = pd.DataFrame(summary_records)

# 통계 요약 생성
summary_stats = summary_df[['Delta Hedge PnL', 'Gamma Hedge PnL']].describe().T
summary_stats = summary_stats.rename(columns={
    'mean': 'Mean',
    'std': 'Std Dev',
    'min': 'Min',
    '25%': 'Q1',
    '50%': 'Median',
    '75%': 'Q3',
    'max': 'Max'
})

# Excel로 저장 (두 시트: PnL, Summary)
with pd.ExcelWriter("hedge_strategy_final_PnL_summary.xlsx", engine='xlsxwriter') as writer:
    summary_df.to_excel(writer, sheet_name="PnL Records", index=False)
    summary_stats.to_excel(writer, sheet_name="Summary Stats")

# %%
from scipy.stats import ttest_rel

# 두 전략의 수익률
delta_pnl = summary_df['Delta Hedge PnL']
gamma_pnl = summary_df['Gamma Hedge PnL']

# Paired t-test
t_stat, p_value = ttest_rel(gamma_pnl, delta_pnl)

print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")

# %%
from statsmodels.stats.descriptivestats import sign_test

stat, p = sign_test(gamma_pnl - delta_pnl)
print(f"Sign test p-value: {p:.4f}")

# %%
#%% Spot Return 계산 (jump date 기준 현물 수익률)
spot_returns = []

for jump_date in jump_times_filtered:
    i0 = data.index.get_loc(jump_date)
    if i0 + T_days >= len(data):
        continue
    S0 = data['Close'].iloc[i0-1]
    S_T = data['Close'].iloc[i0 + T_days]
    ret = (S_T - S0) / S0
    spot_returns.append({
        'Jump Date': jump_date.strftime('%Y-%m-%d'),
        'Spot Return': ret
    })

spot_df = pd.DataFrame(spot_returns)

# summary_df와 병합
merged = pd.merge(summary_df, spot_df, on='Jump Date')

#%% 열 타입 강제 변환: 숫자형(float)으로 변환
cols_to_convert = ['Spot Return', 'Delta Hedge PnL', 'Gamma Hedge PnL']
merged[cols_to_convert] = merged[cols_to_convert].apply(pd.to_numeric, errors='coerce')

comparison_stats = merged[['Spot Return', 'Delta Hedge PnL', 'Gamma Hedge PnL']].agg(['mean', 'std']).T
comparison_stats = comparison_stats.rename(columns={'mean': 'Mean', 'std': 'Std Dev'})
print(comparison_stats)
# %%
#%% jump date 형식 확인 및 통일 (모두 str로 변환)
spot_df['Jump Date'] = spot_df['Jump Date'].astype(str)
summary_df['Jump Date'] = summary_df['Jump Date'].astype(str)

# %%
# 점프 발생 전날 가격 및 수익률 계산
jump_before_returns = []

for jump_date in jump_times_filtered:
    i0 = data.index.get_loc(jump_date)
    if i0 == 0:
        continue
    prev_price = data['Close'].iloc[i0 - 1].item()  # float로 변환
    jump_price = data['Close'].iloc[i0].item()      # float로 변환
    jump_return = (jump_price - prev_price) / prev_price
    jump_before_returns.append({
        'Jump Date': jump_date.strftime('%Y-%m-%d'),
        'Prev Close': prev_price,
        'Jump Day Close': jump_price,
        'Jump Return': jump_return
    })

jump_return_df = pd.DataFrame(jump_before_returns)
jump_return_df.to_excel("jump_day_prev_return.xlsx", index=False)

# %%
from scipy.stats import skew, kurtosis
skewness = skew(gamma_pnl - delta_pnl)
excess_kurtosis = kurtosis(gamma_pnl - delta_pnl, fisher=True)

print(f"Skewness: {skewness:.4f}")
print(f"Excess Kurtosis: {excess_kurtosis:.4f}")
# %%
skewness_d = skew(delta_pnl)
excess_kurtosis_d = kurtosis(delta_pnl, fisher=True)

print(f"Skewness: {skewness_d:.4f}")
print(f"Excess Kurtosis: {excess_kurtosis_d:.4f}")

skewness_g = skew(gamma_pnl)
excess_kurtosis_g = kurtosis(gamma_pnl, fisher=True)

print(f"Skewness: {skewness_g:.4f}")
print(f"Excess Kurtosis: {excess_kurtosis_g:.4f}")
# %%
jb_stat, jb_pvalue = jarque_bera(gamma_pnl - delta_pnl)
print(f"Jarque-Bera test statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}")
# %%
