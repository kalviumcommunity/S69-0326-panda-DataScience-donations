# S69-0326-panda-DataScience-donations

# 📊 Donor Analytics & Fundraising Insights (Sprint 3)
### Team 05 | Problem Statement: Analyzing Non-Profit Donation Data

## 🎯 Project Goal
The objective of this project is to transform raw donation records into actionable insights. By using **Python** and **Data Science** libraries, we aim to help non-profits understand:
* **Donor Retention:** Identifying repeat donors vs. one-time contributors.
* **Frequency Trends:** How often do donors give? (Monthly, Annually, etc.)
* **Fundraising Effectiveness:** Pinpointing "Peak Periods" where donations spike to replicate that success in future campaigns.

## 🛠️ Tech Stack
* **Environment:** Anaconda (Jupyter Notebook / Spyder)
* **Language:** Python 3.x
* **Libraries:** `Pandas` (Data Manipulation), `Matplotlib/Seaborn` (Visualization), `NumPy` (Numerical Analysis).

## 🚀 Analysis Workflow

### 1. Data Cleaning & Preparation
* Handling missing values in donation amounts or dates.
* Converting date strings into `DateTime` objects for time-series analysis.
* Filtering out outliers (extremely small or unusually large "test" donations).

### 2. Behavioral Segmentation
* **Repeat Donor Analysis:** Grouping data by `Donor_ID` to count frequency.
* **Recency, Frequency, Monetary (RFM) Analysis:** Categorizing donors based on how recently they gave and how much they contributed.

### 3. Trend Analysis
* Aggregating donations by month/quarter to identify "High Fundraising Effectiveness" periods.
* Correlation analysis between marketing campaigns and donation spikes.

## 💻 Setup Instructions

To run the analysis locally, ensure you have **Anaconda** installed.

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/kalviumcommunity/S69-0326-panda-DataScience-donations]
