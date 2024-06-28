import streamlit as st
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import os
import pickle

# Function to load and clean data
def load_and_clean_data(file_path):
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date', dayfirst=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Change', 'Volume']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col].str.replace(',', ''), errors='coerce')
        data.fillna(method='ffill', inplace=True)
        return data
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None

# Function to fit and predict with AutoReg model
def fit_and_predict(series, periods):
    model = AutoReg(series, lags=5)
    model_fit = model.fit()
    end_index = len(series) + periods - 1
    predictions = model_fit.predict(start=len(series), end=end_index, dynamic=False)
    return predictions

# Function to predict future data
def predict_future(data, freq, years):
    prediction_period = years * (12 if freq == 'M' else 1)
    predictions = {}
    for col in data.columns:
        predictions[col] = fit_and_predict(data[col], prediction_period)

    future_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(months=1 if freq == 'M' else 1), periods=prediction_period, freq=freq)
    future_data = pd.DataFrame(predictions, index=future_dates)
    return future_data

# Function to add profit/loss column
def add_profit_loss_column(data):
    data['Profit/Loss'] = (data['Close'].diff().fillna(0) > 0).replace({True: 'Profit', False: 'Loss'})
    return data

# Function to identify best investment
def identify_best_investment(data):
    best_month = data[data['Profit/Loss'] == 'Profit']['Close'].idxmax()
    best_volume = data.loc[best_month, 'Volume']
    return best_month, best_volume

# Function to plot data
def plot_data(historical_data, future_data, years_to_predict, freq, plot_type):
    plt.figure(figsize=(12, 6))
    for col in historical_data.columns:
        plt.plot(historical_data.index, historical_data[col], label=f'{col} (Historical)')
        plt.plot(future_data.index, future_data[col], label=f'{col} (Predicted: {years_to_predict} Years)', linestyle='--')
    plt.title(f'Stock Exchange Predictions - {plot_type} Data Predictions for {years_to_predict} Year(s)')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    st.pyplot(plt)

# Function to save data using pickle
def save_data_with_pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# Function to load data using pickle
def load_data_with_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Main Streamlit app
def main():
    st.title('Stock Exchange Predictions')

    # User choice for dataset
    choice = st.radio("Select dataset to predict:", ('KSE 100 index', 'PSX data'))

    if choice == 'KSE 100 index':
        file_path = 'Stock Exchange KSE 100(Pakistan).csv'
    elif choice == 'PSX data':
        file_path = 'PSX.csv.csv'
    else:
        st.error("Invalid choice. Please select a dataset.")
        return

    # Load and clean data
    data = load_and_clean_data(file_path)
    if data is None:
        return

    # Resample the data to monthly and yearly frequencies
    monthly_data = data.resample('M').mean()
    yearly_data = data.resample('Y').mean()

    # Accept input for number of years to predict
    years_to_predict = st.number_input("Enter the number of years to predict:", min_value=1, max_value=10, value=1)

    # Perform predictions
    future_monthly_data = predict_future(monthly_data, 'M', years_to_predict)
    future_yearly_data = predict_future(yearly_data, 'Y', years_to_predict)

    # Add profit/loss column
    future_monthly_data = add_profit_loss_column(future_monthly_data)
    future_yearly_data = add_profit_loss_column(future_yearly_data)

    # Identify best investment
    best_monthly_investment, best_monthly_volume = identify_best_investment(future_monthly_data)

    # Plot the results for monthly predictions
    st.subheader(f'Monthly Data Predictions for {years_to_predict} Year(s)')
    plot_data(monthly_data, future_monthly_data, years_to_predict, 'M', 'Monthly')

    # Plot the results for yearly predictions
    st.subheader(f'Yearly Data Predictions for {years_to_predict} Year(s)')
    plot_data(yearly_data, future_yearly_data, years_to_predict, 'Y', 'Yearly')

    # Save predictions to pickle files
    st.write("Saving predictions with pickle...")
    save_data_with_pickle(future_monthly_data, f'Future_Monthly_Predictions_{years_to_predict}_years.pkl')
    save_data_with_pickle(future_yearly_data, f'Future_Yearly_Predictions_{years_to_predict}_years.pkl')

    # Print predicted values
    st.subheader("Future Monthly Predictions:")
    st.write(future_monthly_data)
    st.subheader("Future Yearly Predictions:")
    st.write(future_yearly_data)

    # Suggest best month and volume to invest
    st.subheader("Best month to invest:")
    st.write(best_monthly_investment.strftime('%Y-%m'))
    st.subheader("Suggested volume to invest in that month:")
    st.write(best_monthly_volume)

if __name__ == '__main__':
    main()
