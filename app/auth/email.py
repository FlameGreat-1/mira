import os
import logging
import smtplib
import ssl
from email.message import EmailMessage
from app.config import config
from app.logger import logger

def send_reset_email(email, reset_link):
    smtp_server = config.SMTP_HOST or "smtp.gmail.com"
    smtp_port = config.SMTP_PORT or 587
    smtp_username = config.SMTP_USERNAME
    smtp_password = config.SMTP_PASSWORD
    sender_email = config.EMAIL_FROM
    
    if not all([smtp_username, smtp_password, email, reset_link]):
        logger.error("Missing required email configuration or parameters")
        return False
    
    try:
        # Create a simple plain text email - no HTML, no templates
        msg = EmailMessage()
        msg['Subject'] = "Reset Your Password"
        msg['From'] = f"OpenAgentFramework <{sender_email}>"
        msg['To'] = email
        
        # Plain text only - no HTML templates
        text_content = f"""
Hello,

We received a request to reset your password for your account.
To complete the process, please click the link below:

{reset_link}

This link will expire in 24 hours for security reasons.

If you didn't request a password reset, please ignore this email or contact support if you have concerns.

Best regards,
The OpenAgentFramework Team
        """
        
        msg.set_content(text_content)
        
        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            
        logger.info(f"Password reset email sent successfully to {email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send password reset email: {str(e)}")
        return False
