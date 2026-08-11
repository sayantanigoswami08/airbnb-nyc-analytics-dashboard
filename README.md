# 🏙️ Airbnb NYC Analytics Dashboard

**Interactive Business Analytics Platform** — Comprehensive market analysis, segmentation, and revenue prediction for Airbnb listings in NYC (2019).

---

## 📊 Dashboard Features

### 1. **📈 Overview**
- Quick KPIs: Total listings, median price, avg reviews, estimated revenue
- Borough breakdown with statistics

### 2. **🔍 EDA & Price Analysis**
- Price distribution (histogram with median line)
- Price by borough (boxplot)
- Room type breakdown (pie chart)
- Engagement metrics by borough

### 3. **🎯 Market Segmentation**
- **K-Means Clustering (k=4)** identifying 4 distinct market segments:
  - 🟢 **Steady Earners**: High availability, mid-price listings
  - 🟡 **Occasional Hosts**: Low availability, casual operators
  - 🔵 **Volume-Driven**: High demand, competitive pricing
  - 🟣 **Niche Premium**: Exclusive, premium offerings

### 4. **💰 Revenue Prediction**
- **Linear Regression Model** (R² = 0.82)
- Top 10 revenue drivers visualization
- Actual vs predicted performance chart
- MAE: ~$1,800/year

### 5. **🎪 Custom Listing Calculator**
- Interactive sliders to predict revenue for any listing configuration
- Real-time calculations
- Detailed insights and recommendations

---

## 🚀 Deployment Instructions

### **Option 1: Deploy to Streamlit Cloud (Recommended ⭐)**

**Step 1: Prepare Your GitHub Repository**

1. Create a new GitHub repository (e.g., `airbnb-nyc-analytics`)
2. Clone it locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/airbnb-nyc-analytics.git
   cd airbnb-nyc-analytics
   ```

3. Copy these files to your repo:
   - `app.py` (main Streamlit app)
   - `requirements.txt` (Python dependencies)
   - `.streamlit/config.toml` (configuration)
   - `AB_NYC_2019.csv` (your dataset) **[IMPORTANT]**
   - `README.md` (this file)

4. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Initial commit: Airbnb NYC Analytics Dashboard"
   git push origin main
   ```

**Step 2: Connect to Streamlit Cloud**

1. Go to [**streamlit.io**](https://streamlit.io)
2. Click **"Deploy an App"** or visit [**share.streamlit.io**](https://share.streamlit.io)
3. Sign in with your **GitHub account**
4. Click **"New app"**
5. Fill in:
   - **Repository:** `YOUR_USERNAME/airbnb-nyc-analytics`
   - **Branch:** `main`
   - **Main file path:** `app.py`
6. Click **"Deploy"** ✅

Your dashboard will be live in ~2-3 minutes!  
**URL:** `https://share.streamlit.io/YOUR_USERNAME/airbnb-nyc-analytics`

---

### **Option 2: Deploy to Render**

1. Sign up at [**render.com**](https://render.com)
2. Create a `render.yaml` in your repo:
   ```yaml
   services:
     - type: web
       name: airbnb-analytics
       env: python
       startCommand: "streamlit run app.py --server.port=10000 --server.address=0.0.0.0"
       buildCommand: "pip install -r requirements.txt"
   ```
3. Connect your GitHub repo
4. Deploy (free tier available)

---

### **Option 3: Deploy to Heroku (Paid)**

1. Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
2. Create a `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
3. Deploy:
   ```bash
   heroku create YOUR_APP_NAME
   git push heroku main
   ```

---

### **Option 4: Deploy to PythonAnywhere (Simple)**

1. Sign up at [**pythonanywhere.com**](https://www.pythonanywhere.com)
2. Upload files via their web interface
3. Create a new web app with Python + Flask/WSGI
4. Follow their Streamlit integration guide

---

## 📋 File Structure

```
airbnb-nyc-analytics/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── AB_NYC_2019.csv                # Dataset (⚠️ REQUIRED)
├── README.md                       # This file
└── .gitignore                      # Git ignore file (optional)
```

---

## ⚙️ Local Development (Test Before Deploying)

### **Setup**

```bash
# Clone your repo (if not done)
git clone https://github.com/YOUR_USERNAME/airbnb-nyc-analytics.git
cd airbnb-nyc-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Run Locally**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Dataset Requirements

The app expects a CSV file named **`AB_NYC_2019.csv`** with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Listing ID |
| `name` | str | Listing title |
| `host_id` | int | Host ID |
| `host_name` | str | Host name |
| `neighbourhood_group` | str | Borough (Manhattan, Brooklyn, etc.) |
| `neighbourhood` | str | Specific neighborhood |
| `latitude` | float | GPS latitude |
| `longitude` | float | GPS longitude |
| `room_type` | str | Entire home/apt, Private room, Shared room |
| `price` | str/float | Price per night |
| `minimum_nights` | int | Minimum stay |
| `number_of_reviews` | int | Total reviews received |
| `reviews_per_month` | float | Reviews per month |
| `last_review` | str | Date of last review |
| `calculated_host_listings_count` | int | Host's other listings |
| `availability_365` | int | Days unavailable per year |

**Download the dataset:** [Kaggle - Airbnb NYC 2019](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)

---

## 🔧 Customization

### **Change Theme**
Edit `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#667eea"          # Change to any hex color
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### **Modify Model Parameters**
In `app.py`, find the clustering section:
```python
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
                          # ↑ Change cluster count here
```

### **Add New Sections**
Add to `st.sidebar.radio()`:
```python
page = st.sidebar.radio("Choose Section:", [
    "📈 Overview",
    "YOUR_NEW_PAGE",    # Add here
    ...
])
```

---

## 📈 Model Performance

- **Clustering**: K-Means (k=4) with 4 distinct market segments
- **Regression**: Linear Regression for revenue prediction
  - **R² Score**: 0.82 (explains 82% of variance)
  - **MAE**: ~$1,800/year (~12% error rate)
  - **Features**: 21 engineered features (pricing, engagement, location, text)

---

## 🎯 Key Insights (from 2019 data)

1. **Manhattan** commands highest median price (~$190/night)
2. **Entire home/apt** listings earn ~2× private rooms
3. **Price & engagement** operate independently (near-zero correlation)
4. **4 distinct segments** with different hosting strategies
5. **High availability** is top positive revenue driver

---

## 🐛 Troubleshooting

### **"CSV not found" Error**
- Ensure `AB_NYC_2019.csv` is in the same directory as `app.py`
- Check filename spelling (case-sensitive on Linux/Mac)

### **Slow loading on Streamlit Cloud**
- The app caches data automatically (@st.cache_data)
- First load may take 30-60 seconds; subsequent loads are instant

### **ModuleNotFoundError**
- Ensure all packages in `requirements.txt` are listed
- Run `pip install -r requirements.txt` locally

### **Port conflicts (local development)**
```bash
streamlit run app.py --server.port 8502
```

---

## 📞 Support & Resources

- **Streamlit Docs:** [docs.streamlit.io](https://docs.streamlit.io)
- **Deployment Guide:** [docs.streamlit.io/deploy](https://docs.streamlit.io/deploy)
- **GitHub Issues:** Report bugs in your repo
- **Kaggle Dataset:** [NYC Airbnb 2019](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)

---

## 📜 License

This project is open-source. Feel free to modify and deploy!

---

## ✨ What's Next?

- [ ] Add real-time data updates
- [ ] Integrate with actual Airbnb API
- [ ] Add geographic heatmaps
- [ ] Implement time-series forecasting
- [ ] Add host profiling dashboard
- [ ] Connect to PostgreSQL for live data

---

**Happy Analytics! 🚀**
