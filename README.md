# ReViveCare - Render Deployment Guide

## 🚀 Quick Deploy to Render

### Prerequisites
- Git repository initialized and connected to GitHub/GitLab
- Render account (free tier works)

### Step 1: Push Your Code to Git

```bash
cd c:\Users\Dell\AndroidStudioProjects\reViveCare\thapar

# Initialize git if not already done
git init

# Add all files
git add .

# Commit changes
git commit -m "Prepare for Render deployment"

# Add your remote repository (replace with your repo URL)
git remote add origin https://github.com/yourusername/revivecare.git

# Push to GitHub
git push -u origin main
```

### Step 2: Create Web Service on Render

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your Git repository
4. Configure the service:
   - **Name**: `revivecare` (or your choice)
   - **Environment**: `Python 3`
   - **Build Command**: `./build.sh`
   - **Start Command**: `cd project/ReviveCare && gunicorn ReviveCare.wsgi:application`
   - **Instance Type**: Free (for testing)

### Step 3: Set Environment Variables

In Render dashboard, go to **Environment** tab and add these variables:

```
SECRET_KEY=generate-a-new-secret-key-here
DEBUG=False
ALLOWED_HOSTS=your-app-name.onrender.com
GROQ_API_KEY=your-groq-api-key
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_SENDER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
DOCTOR_EMAIL=doctor@example.com
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_PHONE_NUMBER=+1234567890
```

To generate a new SECRET_KEY:
```python
from django.core.management.utils import get_random_secret_key
print(get_random_secret_key())
```

### Step 4: Deploy

Click **"Create Web Service"** - Render will automatically:
- Install dependencies from `requirements.txt`
- Run `build.sh` (collect static files, run migrations)
- Start the application with gunicorn

### 🔍 Important Notes

**⚠️ Webcam Features Won't Work**
The exercise tracking features that use `cv2.VideoCapture(0)` require a physical camera and won't work on Render's servers. These features are:
- Side lateral raises tracking
- Bicep curl tracking  
- Other exercise video features

**💾 Database Considerations**
- Currently using SQLite (stored in `db.sqlite3`)
- For production, consider using PostgreSQL (Render offers free PostgreSQL databases)
- SQLite files may be lost when Render restarts your service

**📁 Static Files**
- Static files are served using WhiteNoise
- The `build.sh` script automatically collects static files
- No additional CDN needed for basic deployment

### 🔧 Troubleshooting

**Build Fails**
- Check `build.sh` has Unix line endings (LF, not CRLF)
- Verify all dependencies in `requirements.txt` are compatible
- Check build logs in Render dashboard

**App Crashes on Start**
- Verify all environment variables are set
- Check application logs in Render dashboard
- Ensure `ALLOWED_HOSTS` includes your Render URL

**Static Files Not Loading**
- Make sure `whitenoise` is in `requirements.txt`
- Verify `build.sh` runs `collectstatic`
- Check `settings.py` has WhiteNoise middleware

### 📚 File Structure for Deployment

```
thapar/
├── requirements.txt        ✅ Python dependencies
├── build.sh               ✅ Render build script
├── .gitignore             ✅ Git ignore rules
├── .env.example           ✅ Environment variable template
├── README.md              ✅ This file
└── project/
    └── ReviveCare/
        ├── manage.py      ✅ Django management
        ├── db.sqlite3     ⚠️  Local database
        └── ReviveCare/
            ├── settings.py ✅ Production-ready settings
            └── wsgi.py    ✅ WSGI application
```

### 🎯 Next Steps After Deployment

1. **Test Your Deployment**: Visit your Render URL
2. **Set up PostgreSQL** (recommended):
   - Create a PostgreSQL database on Render
   - Update `DATABASE_URL` environment variable
   - Render will auto-configure if you link the database
3. **Configure Custom Domain** (optional)
4. **Set up monitoring** and error tracking

### 📞 Support

For issues specific to:
- **Render**: https://render.com/docs
- **Django**: https://docs.djangoproject.com/
- **This project**: Check your Git repository issues

---

**Ready to deploy?** Follow the steps above and your app will be live in minutes! 🎉
