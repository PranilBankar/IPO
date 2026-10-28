# 🚀 Deploying IPOPilot AI on Streamlit Community Cloud

This guide provides step-by-step instructions to deploy the IPOPilot web application to **[Streamlit Community Cloud](https://share.streamlit.io)** (100% free hosting for Streamlit apps directly connected to GitHub).

---

## 📋 What was created

1. **`streamlit_app.py`**: A unified, high-performance Streamlit application featuring:
   - 📊 **Live IPO Intelligence Dashboard**: Active issues, GMP, subscription status, AI listing returns, P(loss), and allotment odds.
   - 🔬 **IPO Deep Dive & What-If Studio**: Interactive probability density curve, GMP tracking, and sensitivity simulator with dynamic sliders.
   - 🎯 **Family Capital Optimizer**: Multi-PAN integer programming (OR-Tools CP-SAT) solving portfolio knapsack under SEBI allotment rules.
   - 🧮 **Allotment Odds Calculator**: Mathematical model for Retail (RII) and NII lottery odds.
   - 📜 **Historical Database & Backtest**: 2,287+ past listed Indian IPOs, return distributions, and strategy comparisons.
   - 🧠 **AI Model Card & Architecture**: Deep MDN ensemble specifications and walk-forward validation metrics.
2. **`requirements.txt`**: Complete dependency specification for Streamlit Cloud.
3. **`.streamlit/config.toml`**: Custom high-end dark slate & blue theme.
4. **`scripts/start_streamlit.ps1`**: 1-click local launch script for Windows.

---

## 🏃 Step 1: Run & Test Locally

Before deploying to the cloud, verify the app runs locally:

```powershell
# Using the helper script:
./scripts/start_streamlit.ps1

# OR directly with python:
python -m streamlit run streamlit_app.py
```

The app will open automatically at **`http://localhost:8501`**.

---

## 🌐 Step 2: Push Changes to GitHub

Commit the new Streamlit files, models, and database to your GitHub repository:

```powershell
# 1. Review status
git status

# 2. Add Streamlit files, configuration, models, and database
git add .streamlit/ requirements.txt streamlit_app.py scripts/start_streamlit.ps1 STREAMLIT_DEPLOYMENT.md .gitignore
git add data/models/ data/ipopilot.db

# 3. Commit
git commit -m "Add Streamlit web app and deployment configuration"

# 4. Push to your main branch
git push origin main
```

> [!TIP]
> The trained model (`members.pt` ~2.4MB) and SQLite database (`ipopilot.db` ~20MB) are well below GitHub's 100MB file limit. Pushing them ensures Streamlit Community Cloud starts with all 2,287 past IPOs and the pre-trained Deep MDN model loaded out of the box!

---

## ☁️ Step 3: Deploy to Streamlit Community Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
2. Click the **"New app"** button (top right).
3. Fill in the deployment form:
   - **Repository:** `PranilBankar/IPO` (or your GitHub repo)
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL (optional):** Choose a custom subdomain (e.g. `ipopilot-ai.streamlit.app`)
4. Click **"Advanced settings..."**:
   - **Python version:** Select **`3.10`** or **`3.11`**.
   - **Secrets (optional):** If you want to specify environment variables, you can paste:
     ```toml
     IPOPILOT_ENABLE_SCHEDULER = "0"
     ```
5. Click **"Deploy!"** 🚀

Streamlit Cloud will provision the container, install packages from `requirements.txt`, and launch your app. In about 2–3 minutes, your live public URL will be ready to share!

---

## 🔧 Cloud Considerations & Performance Tips

- **Memory Limits:** Streamlit Community Cloud offers 1GB RAM on free instances. The app is optimized to keep memory footprints low (~350MB).
- **Auto-Reboot:** If an unhandled exception occurs on a freshly cloned instance without a database, the app automatically initializes `init_db()` and seeds fictional demo issues to guarantee the UI never crashes.
- **Continuous Deployment:** Any future commits you push to `main` on GitHub will automatically trigger a redeploy on Streamlit Community Cloud.
