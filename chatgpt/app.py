import os
import json
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("SMTP_SERVER", "smtp.gmail.com")
os.environ.setdefault("SMTP_PORT", "587")
os.environ.setdefault("EMAIL_SENDER", "")
os.environ.setdefault("EMAIL_PASSWORD", "")
os.environ.setdefault("DOCTOR_EMAIL", "")

SYSTEM_PROMPT = """You are ReViveCare, an AI-powered post-discharge recovery support chatbot.
Your role is to support patients emotionally and informationally during recovery after surgery or injury.
You are NOT a doctor and you MUST NOT diagnose, prescribe medication, or alter treatment plans.

You will be given full patient context provided by their doctor.
Use this context to answer user queries accurately, calmly, and empathetically.

Your goals:
1. Reassure and guide the patient using simple, non-alarming language
2. Compare their reported symptoms against their specific medical context
3. Determine if symptoms align with expected recovery for THEIR condition
4. Encourage appropriate next steps without diagnosing
5. Detect potentially serious situations based on THEIR specific medical context

IMPORTANT RULES:
- Never provide a diagnosis
- Never prescribe or change medications
- Never claim certainty about outcomes
- Never dismiss the user's concern
- Be calm, supportive, and empathetic
- Base all assessments on the patient's specific condition and context provided by their doctor

You must return EXACTLY a valid JSON object for every user message with these keys:
1. patient_response: A clear, supportive answer to the user's query based on their specific medical context
2. seriousness_score: A single floating-point number between 0.0 and 1.0

SERIOUSNESS SCORING GUIDELINES:
0.00 – 0.30 → Symptoms align with expected recovery for their specific condition, reassurance sufficient
0.31 – 0.60 → Monitor closely, routine follow-up advised based on their condition
0.61 – 0.75 → Concerning for their specific condition, recommend contacting doctor soon
> 0.75 → Potential red flag for their specific condition, immediate escalation required

When scoring seriousness:
- Consider the patient's specific condition and what is normal for THEIR recovery
- Compare symptoms against expected vs. concerning signs for THEIR medical context
- Account for their prescribed medications and potential side effects
- Consider their recovery timeline and stage

If seriousness_score > 0.75:
- In your patient_response, calmly advise the patient to contact their doctor or hospital immediately
- Do NOT use panic-inducing language unless absolutely necessary
- Reference their specific condition when explaining why they should seek help

STRICT OUTPUT FORMAT (valid JSON only, no extra text):
{{
  "patient_response": "your response here",
  "seriousness_score": 0.0
}}

PATIENT MEDICAL CONTEXT:
{patient_context}
"""

def initialize_model():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
        timeout=30,
        max_retries=2,
    )

def parse_response(raw_response):
    try:
        response_text = raw_response.content.strip()
        
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            patient_response = parsed.get("patient_response", "")
            seriousness_score = float(parsed.get("seriousness_score", 0.0))
            
            seriousness_score = max(0.0, min(1.0, seriousness_score))
            
            return patient_response, seriousness_score
        else:
            return "I apologize, but I'm having trouble processing your request. Please try rephrasing your question.", 0.0
            
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"Parsing error: {e}")
        return "I apologize, but I'm having trouble processing your request. Please try rephrasing your question.", 0.0

def send_email_alert(patient_context, conversation_history, patient_response, seriousness_score, user_query):
    try:
        smtp_server = os.environ.get("SMTP_SERVER")
        smtp_port = int(os.environ.get("SMTP_PORT"))
        sender_email = os.environ.get("EMAIL_SENDER")
        sender_password = os.environ.get("EMAIL_PASSWORD")
        doctor_email = os.environ.get("DOCTOR_EMAIL")
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = doctor_email
        msg['Subject'] = f"URGENT - Patient Alert from ReViveCare (Score: {seriousness_score:.2f})"
        
        recent_history = ""
        for i, message in enumerate(conversation_history[-6:]):
            if isinstance(message, HumanMessage):
                recent_history += f"Patient: {message.content}\n"
            elif isinstance(message, AIMessage):
                recent_history += f"ReViveCare: {message.content}\n"
            recent_history += "\n"
        
        email_body = f"""
URGENT PATIENT ALERT - ReViveCare System
============================================

Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Seriousness Score: {seriousness_score:.2f}

LATEST PATIENT QUERY:
{user_query}

SYSTEM RESPONSE TO PATIENT:
{patient_response}

RECENT CONVERSATION HISTORY:
{recent_history}

PATIENT MEDICAL CONTEXT:
{patient_context}

RECOMMENDATION:
Please contact the patient immediately to assess their condition and provide appropriate medical guidance.

This is an automated alert from the ReViveCare post-discharge monitoring system.
"""
        
        msg.attach(MIMEText(email_body, 'plain'))
        
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, doctor_email, text)
        server.quit()
        
        print("\n" + "="*60)
        print("🚨 ALERT: HIGH SERIOUSNESS SCORE DETECTED")
        print("="*60)
        print(f"Seriousness Score: {seriousness_score:.2f}")
        print(f"Current Patient Query: {user_query}")
        print(f"Bot Response: {patient_response}")
        print(f"\n✅ EMAIL ALERT SENT SUCCESSFULLY TO: {doctor_email}")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print("🚨 ALERT: HIGH SERIOUSNESS SCORE DETECTED")
        print("="*60)
        print(f"Seriousness Score: {seriousness_score:.2f}")
        print(f"Current Patient Query: {user_query}")
        print(f"Bot Response: {patient_response}")
        print(f"\n⚠️ EMAIL SEND FAILED: {str(e)}")
        print("\n📧 MOCK EMAIL CONTENT:")
        print("Subject: URGENT - Patient Alert from ReViveCare")
        print("Body:")
        print(f"  A patient has reported concerning symptoms.")
        print(f"  Seriousness Score: {seriousness_score:.2f}")
        print(f"  Latest Query: '{user_query}'")
        print(f"  System Response: '{patient_response}'")
        print(f"\n  Recent Conversation History:")
        for i, msg in enumerate(conversation_history[-6:]):
            if isinstance(msg, HumanMessage):
                print(f"    Patient: {msg.content[:100]}...")
            elif isinstance(msg, AIMessage):
                print(f"    ReViveCare: {msg.content[:100]}...")
        print(f"\n  Patient Context: {patient_context[:200]}...")
        print(f"  Recommendation: Please contact patient immediately.")
        print("="*60 + "\n")

def main():
    llm = initialize_model()
    
    print("="*60)
    print("REVIVECARE - Post-Discharge Recovery Support System")
    print("="*60)
    
    print("\nPlease provide the patient medical context from the doctor:")
    print("Include all relevant information such as:")
    print("- Patient name and condition/surgery type")
    print("- Date of surgery/injury and recovery stage")
    print("- Prescribed medications with dosages")
    print("- Assigned exercises and restrictions")
    print("- Expected recovery symptoms")
    print("- Warning signs specific to their condition")
    print("\nType 'END' on a new line when finished:\n")
    
    
    patient_context = """Patient Name: Rahul Sharma
Age: 32

Primary Condition:
- Right knee ACL reconstruction surgery

Surgery Details:
- Arthroscopic ACL reconstruction using hamstring graft
- Surgery date: 3 weeks ago

Current Recovery Stage:
- Early rehabilitation phase (Week 3 post-op)

Prescribed Medications:
- Paracetamol 500 mg as needed for pain (max 3 times/day)
- Etoricoxib 60 mg once daily for inflammation (for 10 days post-op, now stopped)
- No antibiotics currently

Assigned Physiotherapy & Exercises:
- Quad sets (3 sets of 10 reps daily)
- Straight leg raises (3 sets of 10 reps daily)
- Heel slides (gentle range-of-motion exercise)
- Standing weight shifting exercises
- Avoid running, jumping, or twisting movements

Post-Operative Care Instructions:
- Mild pain, stiffness, and swelling after exercises is expected
- Knee stiffness in the morning is common
- Use ice packs after physiotherapy if swelling increases
- Keep surgical wound clean and dry
- Gradual improvement in range of motion expected over weeks

Expected Normal Symptoms:
- Mild to moderate pain after exercises
- Temporary swelling around the knee
- Muscle tightness or soreness
- Slight warmth around surgical area

Warning Signs (Red Flags):
- Fever above 38°C
- Increasing redness, pus, or foul-smelling discharge from surgical site
- Sudden severe pain not relieved by rest
- Knee locking or giving way
- Severe calf pain or swelling
- Breathlessness or chest pain
"""
    
    if not patient_context:
        print("No patient context provided. Exiting.")
        return
    
    system_prompt_with_context = SYSTEM_PROMPT.format(patient_context=patient_context)
    
    print("\n" + "="*60)
    print("Patient Context Loaded Successfully")
    print("="*60)
    print("\nYou can now start asking questions.")
    print("The system will maintain conversation history and context.")
    print("Type 'exit' or 'quit' to end the session.\n")
    print("="*60 + "\n")
    
    conversation_history = []
    
    while True:
        user_input = input("Patient: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nThank you for using ReViveCare. Wishing you a smooth recovery!")
            break
        
        if not user_input:
            print("Please enter a valid question.\n")
            continue
        
        messages = [SystemMessage(content=system_prompt_with_context)]
        
        messages.extend(conversation_history)
        
        current_query = HumanMessage(content=user_input)
        messages.append(current_query)
        
        try:
            response = llm.invoke(messages)
            patient_response, seriousness_score = parse_response(response)
            
            print(f"\nReViveCare: {patient_response}")
            print(f"[Internal Seriousness Score: {seriousness_score:.2f}]\n")
            
            conversation_history.append(HumanMessage(content=user_input))
            conversation_history.append(AIMessage(content=patient_response))
            
            if seriousness_score > 0.75:
                send_email_alert(patient_context, conversation_history, patient_response, seriousness_score, user_input)
            
        except Exception as e:
            print(f"\nError communicating with the AI model: {e}")
            print("Please try again or contact support if the issue persists.\n")

if __name__ == "__main__":
    main()