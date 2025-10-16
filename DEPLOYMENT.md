# ArsiDet Deployment Guide

This guide explains how to deploy ArsiDet (Arabic Sign Language Detection) to various cloud platforms with **live camera support**.

## How It Works

ArsiDet now supports **THREE camera modes**:

1. **WebRTC Live Mode** (🌐 **RECOMMENDED for Cloud**) - Real-time live video streaming using WebRTC. Works on ALL cloud platforms!
2. **Browser Camera Snapshot Mode** (📸 Works on Cloud) - Uses Streamlit's `st.camera_input()` to capture individual images
3. **Local Webcam Mode** (🎥 Local Only) - Uses OpenCV to access the server's webcam (only works locally)

For cloud deployment, **WebRTC Live Mode is the best choice** as it provides real-time continuous video processing just like running locally!

---

## Deployment Options

### Option 1: Render (Recommended)

**Steps:**

1. Push your code to GitHub

2. Go to [render.com](https://render.com) and create a new Web Service

3. Connect your GitHub repository

4. Use these settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements-deploy.txt`
   - **Start Command**: `streamlit run main.py --server.port=$PORT --server.address=0.0.0.0`
   - **Python Version**: 3.12.0

5. Add environment variable:
   - **Key**: `OPENAI_API_KEY`
   - **Value**: Your OpenAI API key (optional, users can also enter it in the app)

6. Deploy!

**Alternatively**, you can use the included `render.yaml`:
```bash
# Just push to GitHub and Render will auto-detect the render.yaml
git add .
git commit -m "Ready for deployment"
git push
```

---

### Option 2: Streamlit Community Cloud (Easiest)

**Steps:**

1. Push your code to GitHub

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Click "New app"

4. Select your repository, branch (main), and main file ([main.py](main.py))

5. Click "Advanced settings" and add secrets:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

6. Click "Deploy"!

**Pros:**
- Completely free
- Optimized for Streamlit apps
- Auto-redeploys on git push
- Built-in secrets management

---

### Option 3: Hugging Face Spaces

**Steps:**

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)

2. Choose "Streamlit" as the SDK

3. Upload your files or connect your GitHub repo

4. Create a `.env` file in Settings > Repository secrets:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

5. The app will auto-deploy!

**Pros:**
- Free GPU support
- Great for ML apps
- Good community

---

### Option 4: Railway

**Steps:**

1. Go to [railway.app](https://railway.app)

2. Create a new project from GitHub repo

3. Railway will auto-detect Python

4. Add environment variable `OPENAI_API_KEY`

5. Deploy!

**Note:** Railway charges after free tier ($5/month credit)

---

## Pre-Deployment Checklist

- [ ] Code is pushed to GitHub
- [ ] `.env` file is in `.gitignore` (don't commit API keys!)
- [ ] Model files are in the `models/` directory
- [ ] [requirements-deploy.txt](requirements-deploy.txt) is up to date
- [ ] [render.yaml](render.yaml) configured (if using Render)

---

## Using the Deployed App

Once deployed, users can:

1. **Select "WebRTC Live" mode** (RECOMMENDED - default for cloud)
2. **Click "START"** to begin live video streaming
3. Allow camera access when prompted by browser
4. The app will continuously detect signs in real-time
5. Detected signs are buffered every 2 seconds
6. AI interprets the complete sentence every 20 seconds (if API key is provided)

**Alternative**: Use "Browser Camera Snapshot" mode to capture individual images instead of continuous video.

---

## Troubleshooting

### Camera not working on deployment
- Make sure you selected **"Browser Camera"** mode (not "Local Webcam")
- Check browser permissions for camera access
- Try HTTPS (required for camera access in browsers)

### Large slug size / Slow build
- The PyTorch and CUDA dependencies are large (~2GB)
- Consider using CPU-only PyTorch for cloud deployment:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

### Out of memory errors
- Reduce model size or use a smaller YOLO model
- Upgrade to a paid tier with more RAM
- Use CPU instead of GPU

### Model file not found
- Ensure `models/ArabicSignLanguage60.pt` is in your repo
- Check the path in [src/utils/config.py](src/utils/config.py:14)

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Optional | OpenAI API key for AI interpretation (users can also enter in UI) |
| `PORT` | Auto-set | Port for the web server (set by platform) |

---

## Cost Estimates

| Platform | Free Tier | Paid Tier |
|----------|-----------|-----------|
| **Streamlit Cloud** | ✅ Free forever | N/A |
| **Hugging Face** | ✅ Free forever | Paid for private spaces |
| **Render** | ✅ Free (sleeps after inactivity) | $7/month |
| **Railway** | $5 credit/month | Pay as you go |

---

## Recommended Deployment

For most users, I recommend **Streamlit Community Cloud**:
- Completely free
- Easy setup
- Optimized for Streamlit
- Auto SSL/HTTPS
- Fast deployments

---

## Need Help?

- Check the [Streamlit deployment docs](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)
- Check the [Render deployment docs](https://render.com/docs/deploy-streamlit)
- Open an issue on GitHub

---

## Local Development

To run locally with live camera:

```bash
# Install dependencies
uv sync

# Run the app
uv run streamlit run main.py
```

Then select **"Local Webcam"** mode for continuous video feed.
