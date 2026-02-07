import smtplib
from email.mime.text import MIMEText

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

EMAIL_SENDER = "animeguy055@gmail.com"          # must be real Gmail
EMAIL_PASSWORD = "zuby ghtn xfbt itco"  # NOT Gmail password
EMAIL_RECEIVER = "sarthak.molu08@gmail.com"         # can be same or different

msg = MIMEText("This is a test email from ReViveCare.\n\nIf you received this, SMTP is working.")
msg["Subject"] = "ReViveCare SMTP Test"
msg["From"] = EMAIL_SENDER
msg["To"] = EMAIL_RECEIVER

try:
    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(EMAIL_SENDER, EMAIL_PASSWORD)
    server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
    server.quit()
    print("✅ Email sent successfully!")

except Exception as e:
    print("❌ Email failed:")
    print(e)
