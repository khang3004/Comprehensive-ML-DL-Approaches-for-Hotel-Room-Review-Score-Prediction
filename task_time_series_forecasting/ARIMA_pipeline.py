# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
from pmdarima import auto_arima
import os
import logging
from datetime import datetime
import joblib  # Thêm import joblib để lưu/load model
import argparse  # Thêm argument parser

warnings.filterwarnings("ignore")

# Load data và xử lý index
file_path = './Health_US.csv'
data = pd.read_csv(file_path)
data['date'] = pd.to_datetime(data['date'])

# Xử lý ngày trùng lặp và sắp xếp
data = data.drop_duplicates(subset='date', keep='first')

# Tạo index đầy đủ từ ngày đầu đến ngày cuối
full_date_range = pd.date_range(start=data['date'].min(),
                               end=data['date'].max(),
                               freq='W-MON')  # Weekly frequency starting Monday

# Reindex data với đầy đủ ngày
data.set_index('date', inplace=True)
data = data.reindex(full_date_range)

# Lấy dữ liệu OT và xử lý missing values
y = data['OT']
y = y.astype(float)

# Xử lý missing values theo thứ tự
y = y.interpolate(method='linear', limit_direction='both')  # Nội suy tuyến tính
y = y.fillna(method='ffill')  # Forward fill cho các giá trị đầu
y = y.fillna(method='bfill')  # Backward fill cho các giá trị cuối
y = y.fillna(y.mean())  # Điền giá trị trung bình cho bất kỳ NA nào còn lại

# Kiểm tra lại missing values
print("\nKiểm tra missing values sau khi xử lý:")
print(f"Số lượng missing values: {y.isna().sum()}")

# Nếu vẫn còn missing values, loại bỏ chúng (không khuyến khích nhưng là giải pháp cuối cùng)
if y.isna().sum() > 0:
    y = y.dropna()
    print(f"Đã loại bỏ missing values. Số lượng điểm dữ liệu còn lại: {len(y)}")

# Tiếp tục với phân tích mùa vụ
decomposition = sm.tsa.seasonal_decompose(y, 
                                        model='additive',
                                        period=52)

# 1. Time Series Exploratory Data Analysis - EDA
## Visualize line plot
y.plot(figsize=(10, 6))
plt.xlabel('Date')
plt.ylabel('Influenza Patients Proportion')
plt.title('Influenza Patients Proportion Over Time')
plt.show()

## Time Series Decomposition
decomposition.plot()
plt.show()


# Tạo thư mục results nếu chưa tồn tại
results_dir = './results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Thiết lập logging với encoding='utf-8'
log_filename = f'results/arima_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Hàm helper để lưu biểu đồ
def save_plot(plt, filename):
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()

# 1. Time Series Exploratory Data Analysis - EDA
## Visualize line plot
plt.figure(figsize=(12, 6))
y.plot()
plt.xlabel('Date')
plt.ylabel('Influenza Patients Proportion')
plt.title('Influenza Patients Proportion Over Time')
save_plot(plt, 'time_series_plot.png')

## Time Series Decomposition
decomposition = sm.tsa.seasonal_decompose(y, model='additive', period=52)
decomposition.plot()
save_plot(plt, 'decomposition.png')

## ADF and KPSS Tests for Stationarity
adf_result = adfuller(y)
print(f'ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}')
kpss_result = kpss(y)
print(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

# Log kết quả kiểm định
logging.info("Kết quả kiểm định tính dừng:")
logging.info(f'ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}')
logging.info(f'KPSS Statistic: {kpss_result[0]}, p-value: {kpss_result[1]}')

## Differencing the data
diff_y = y.copy()
adf_p_value = adf_result[1]
iteration = 0
while adf_p_value > 0.05 and iteration < 3:  # Thêm giới hạn số lần lặp
    diff_y = diff_y.diff().dropna()
    adf_result = adfuller(diff_y)
    adf_p_value = adf_result[1]
    iteration += 1
    print(f'Iteration {iteration}: ADF p-value = {adf_p_value}')

## Plot ACF and PACF
plot_acf(y, lags=40)
plt.title('ACF Before Differencing')
plt.show()

plot_pacf(y, lags=40)
plt.title('PACF Before Differencing')
plt.show()

plot_acf(diff_y, lags=40)
plt.title('ACF After Differencing')
plt.show()

plot_pacf(diff_y, lags=40)
plt.title('PACF After Differencing')
plt.show()

## STL Decomposition
# Sử dụng period=52 cho dữ liệu hàng tuần (52 tuần/năm)
stl = STL(y, period=52)
result = stl.fit()
result.plot()
plt.show()

## Rolling Statistics
window_size = 52  # Một năm
rolling_mean = y.rolling(window=window_size).mean()
rolling_std = y.rolling(window=window_size).std()

plt.figure(figsize=(12, 6))
plt.plot(y, label='Original')
plt.plot(rolling_mean, color='red', label='Rolling Mean')
plt.plot(rolling_std, color='black', label='Rolling Std')
plt.legend()
plt.title('Rolling Statistics')
plt.show()

## Histogram and Distribution Plot
plt.figure(figsize=(10, 6))
y.hist(bins=30)
plt.title('Histogram of Values')
plt.show()

# Kiểm định tính dừng trên dữ liệu gốc
print("\nKiểm định tính dừng trên dữ liệu gốc:")
print("-" * 50)
adf_test = adfuller(y)
print(f"ADF Statistic: {adf_test[0]}, p-value: {adf_test[1]}")

kpss_test = kpss(y, regression='c', nlags="auto")
print(f"KPSS Statistic: {kpss_test[0]}, p-value: {kpss_test[1]}")

# Lấy sai phân bậc 1
y_diff = np.diff(y)

# Kiểm định tính dừng sau khi lấy sai phân
print("\nKiểm định tính dừng sau khi lấy sai phân bậc 1:")
print("-" * 50)
adf_test_diff = adfuller(y_diff)
print(f"ADF Statistic: {adf_test_diff[0]}, p-value: {adf_test_diff[1]}")

kpss_test_diff = kpss(y_diff, regression='c', nlags="auto")
print(f"KPSS Statistic: {kpss_test_diff[0]}, p-value: {kpss_test_diff[1]}")

# 2. Model Training
## Split data
train_size = int(len(y) * 0.7)
train_data = y[:train_size]
test_data = y[train_size:]

## Giảm kích thước dữ liệu bằng cách lấy mẫu theo tháng
y_monthly = y.resample('M').mean()

# Hoặc lấy dữ liệu gần đây hơn (ví dụ: 5 năm gần nhất)
recent_years = 5
y_recent = y[-52*recent_years:]  # 52 tuần * số năm

# Thêm argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--load_model', action='store_true', help='Load pretrained model instead of training')
args = parser.parse_args()

# Đường dẫn để lưu model
model_path = os.path.join(results_dir, 'best_arima_model.pkl')

if args.load_model:
    # Loading mode
    if os.path.exists(model_path):
        print("Đang load model đã train...")
        model_info = joblib.load(model_path)
        auto_model = model_info['model']
        logging.info(f"Đã load model được train vào: {model_info['training_date']}")
        logging.info(f"AIC: {model_info['aic']}")
        logging.info(f"BIC: {model_info['bic']}")
    else:
        raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")

# Split data và fit model cho cả trường hợp train mới và load model
train_size = int(len(y) * 0.7)
train_data = y[:train_size]
test_data = y[train_size:]

# Fit model với tham số tối ưu
optimal_model = ARIMA(train_data, 
                     order=(auto_model.order[0], 
                           auto_model.order[1], 
                           auto_model.order[2]))
optimal_model_fit = optimal_model.fit()

# Tính predictions cho tập train
train_predictions = optimal_model_fit.get_prediction(start=0).predicted_mean

# Tính metrics cho tập train
train_mse = np.mean((train_data.values - train_predictions) ** 2)
train_rmse = np.sqrt(train_mse)
train_mae = np.mean(np.abs(train_data.values - train_predictions))
train_mape = np.mean(np.abs((train_data.values - train_predictions) / train_data.values)) * 100

print('\nTrain Metrics:')
print(f'MSE: {train_mse:.4f}')
print(f'RMSE: {train_rmse:.4f}')
print(f'MAE: {train_mae:.4f}')
print(f'MAPE: {train_mape:.4f}%')

# Expanding Window Forecast cho test set
predictions = []
history = train_data.values
test_index = test_data.index

for t in range(len(test_data)):
    model = ARIMA(history, order=auto_model.order)
    model_fit = model.fit()
    yhat = model_fit.forecast(steps=1)[0]
    predictions.append(yhat)
    history = np.append(history, test_data.iloc[t])

predictions = np.array(predictions)

# Tính metrics cho tập test
test_mse = np.mean((test_data.values - predictions) ** 2)
test_rmse = np.sqrt(test_mse)
test_mae = np.mean(np.abs(test_data.values - predictions))
test_mape = np.mean(np.abs((test_data.values - predictions) / test_data.values)) * 100

print('\nTest Metrics:')
print(f'MSE: {test_mse:.4f}')
print(f'RMSE: {test_rmse:.4f}')
print(f'MAE: {test_mae:.4f}')
print(f'MAPE: {test_mape:.4f}%')

# Log kết quả
logging.info("\nMetrics trên tập train:")
logging.info(f"RMSE: {train_rmse:.4f}")
logging.info(f"MAE: {train_mae:.4f}")
logging.info(f"MAPE: {train_mape:.4f}%")

logging.info("\nMetrics trên tập test:")
logging.info(f"RMSE: {test_rmse:.4f}")
logging.info(f"MAE: {test_mae:.4f}")
logging.info(f"MAPE: {test_mape:.4f}%")

# Cập nhật metrics vào model_info
model_info.update({
    'train_metrics': {
        'rmse': train_rmse,
        'mae': train_mae,
        'mape': train_mape
    },
    'test_metrics': {
        'rmse': test_rmse,
        'mae': test_mae,
        'mape': test_mape
    }
})

# Vẽ biểu đồ dự đoán vs thực tế
plt.figure(figsize=(12, 6))
plt.plot(test_index, test_data.values, label='Actual', color='blue')
plt.plot(test_index, predictions, label='Predicted', color='red')
plt.title('ARIMA: Actual vs Predicted Values')
plt.xlabel('Date')
plt.ylabel('Influenza Patients Proportion')
plt.legend()
save_plot(plt, 'predictions_vs_actual.png')

# Phân tích residuals
residuals = pd.Series(optimal_model_fit.resid, index=optimal_model_fit.fittedvalues.index)

# Plot residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals)
plt.title('Model Residuals')
plt.xlabel('Date')
plt.ylabel('Residual Value')
save_plot(plt, 'residuals.png')

# Histogram của residuals
plt.figure(figsize=(12, 6))
residuals.hist(bins=30)
plt.title('Residuals Distribution')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
save_plot(plt, 'residuals_distribution.png')

# Q-Q Plot
plt.figure(figsize=(12, 6))
sm.graphics.qqplot(residuals, line='45')
plt.title('Q-Q Plot of Residuals')
save_plot(plt, 'residuals_qq_plot.png')

# ACF của residuals
plt.figure(figsize=(12, 6))
plot_acf(residuals)
plt.title('ACF of Residuals')
save_plot(plt, 'residuals_acf.png')

# Ljung-Box test
lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
logging.info("\nLjung-Box Test Results:")
logging.info(str(lb_test))

# Scatter plot của residuals vs fitted values
plt.figure(figsize=(12, 6))
plt.scatter(optimal_model_fit.fittedvalues, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.axhline(y=0, color='r', linestyle='--')
save_plot(plt, 'residuals_vs_fitted.png')

# Lưu lại model_info nếu đang trong chế độ training
if not args.load_model:
    joblib.dump(model_info, model_path)
    logging.info(f"Đã cập nhật metrics và lưu các biểu đồ đánh giá vào: {results_dir}")
