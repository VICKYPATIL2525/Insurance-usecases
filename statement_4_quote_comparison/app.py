'''Flask Web Application for Quote Comparison Chatbot
Wraps the main.py conversational logic into a web interface'''

import json
import os
from flask import Flask, request, jsonify, render_template, session
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- Load env ----------------
load_dotenv()

# ---------------- Initialize Flask App ----------------
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

# ---------------- Setup (from main.py) ----------------
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma(
    persist_directory="../shared/chroma_store",
    embedding_function=embeddings,
    collection_name="insurance_quotes"
)

# LLM1: Classifies user input
llm1 = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini", temperature=0.0, max_tokens=300,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# LLM2: Generates comparison answers
llm2 = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini", temperature=0.2, max_tokens=800,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------- Core Logic (from main.py) ----------------
def understand_question(question, prev_questions):
    """LLM1: Classify question as NEW/FOLLOW_UP/INVALID and extract premiums"""

    if prev_questions:
        prev_context = ""
        for q in prev_questions:
            prev_context += f"- {q}\n"
    else:
        prev_context = "None"

    system_msg = SystemMessage(content="""You are a query classifier for an insurance comparison chatbot.

Your job: Analyze user questions and extract premium amounts.

Classification rules:
1. NEW: User provides 2-3 premium amounts to compare
   - Must have at least 2 numbers
   - Can be any format: "18000, 22500" or "compare 18k and 22.5k" or "18000 vs 22500"

2. FOLLOW_UP: User asks about previously shown results
   - Questions like "which is cheaper?", "which has no deductible?", "best for family of 4?"
   - NO new premium amounts mentioned

3. INVALID:
   - Only 1 premium amount (need at least 2)
   - No premium amounts at all
   - Not insurance related (like "what is weather?")
   - More than 3 premiums

Output format (JSON only):
{
  "question_type": "NEW" | "FOLLOW_UP" | "INVALID",
  "premium_amounts": [list of numbers] or null,
  "reason": "brief explanation"
}

Examples:
Input: "compare 18000, 22500, 28000"
Output: {"question_type": "NEW", "premium_amounts": [18000, 22500, 28000], "reason": "3 premiums to compare"}

Input: "which one is cheaper?"
Output: {"question_type": "FOLLOW_UP", "premium_amounts": null, "reason": "Asking about previous results"}

Input: "show me 18000 plan"
Output: {"question_type": "INVALID", "premium_amounts": [18000], "reason": "Only 1 premium, need at least 2"}

Input: "what is the weather?"
Output: {"question_type": "INVALID", "premium_amounts": null, "reason": "Not insurance related"}""")

    human_msg = HumanMessage(
        content=f"""Previous questions asked by user:{prev_context} Current question: "{question}"
        Classify this question and extract premium amounts."""
        )

    response = llm1.invoke([system_msg, human_msg])

    try:
        return json.loads(response.content)
    except:
        return {"question_type": "INVALID", "premium_amounts": None, "reason": "Error parsing AI response"}


def get_plans(premiums):
    """Search vector DB for plans matching the given premium amounts"""

    plans = []
    for premium in premiums:
        results = vector_db.similarity_search(
            query=f"insurance plan {premium}",
            k=1,
            filter={"premium": premium}
        )
        if results:
            plans.append(results[0])

    return plans


def generate_answer(question, plans, history):
    """LLM2: Generate insurance plan comparison using conversation history"""

    if plans:
        plan_text = ""
        for plan in plans:
            plan_text += plan.page_content + "\n\n"
    else:
        plan_text = "No plans available"

    messages = []
    sys_content = """You are an expert insurance advisor who explains insurance plans in simple, easy-to-understand language.

Your job: Compare insurance plans and help users make informed decisions.

What to focus on:
1. Premium (yearly cost) - Which is cheapest? Most expensive? Price differences?
2. Sum Insured (coverage amount) - How much coverage do you get?
3. Deductible (what you pay first) - Zero deductible is better. High deductible = you pay more when claiming
4. Family Size (max members covered) - Important for families
5. Waiting Periods - When coverage starts for certain conditions
6. Room Rent Limits - Single AC, shared, ICU coverage
7. Co-payment - Do you have to pay a % when claiming?

How to explain:
- Use simple language (avoid jargon)
- Use bullet points and tables for clarity
- Explain trade-offs clearly
  Example: "Plan A is ₹4,500 cheaper per year BUT has a ₹25,000 deductible (you pay first ₹25k of any claim)"
- Make specific recommendations based on user needs
  Example: "For a couple with no kids, Plan B is better because..."
- Use comparisons: "Plan A vs Plan B: Both cover family of 4, but Plan B has zero deductible"

What NOT to do:
- Don't make up information
- Don't mention plans that aren't provided
- Don't use complex insurance terms without explaining them
- Don't be vague - give specific numbers and reasons

Format:
- Start with a quick summary table
- Then detailed comparison
- End with recommendation based on their question"""

    system_msg = SystemMessage(content=sys_content)
    messages.append(system_msg)

    if history:
        for item in history:
            messages.append(HumanMessage(content=item['user']))
            messages.append(AIMessage(content=item['bot']))

    current_question = HumanMessage(content=f"""Here are the insurance plans to compare:{plan_text}
    User's question: {question}
    Please compare these plans and answer the user's question in simple, clear terms.""")
    messages.append(current_question)

    response = llm2.invoke(messages)
    return response.content


def chat(question, history, prev_questions):
    """Main chat function - routes questions and manages conversation state"""

    understanding = understand_question(question, prev_questions)
    q_type = understanding["question_type"]
    premiums = understanding["premium_amounts"]

    if q_type == "INVALID":
        return (
            f"Please provide 2-3 premium amounts to compare.\n"
            f"Example: 'compare 18000, 22500, 28000'\n\n"
            f"Reason: {understanding['reason']}",
            history,
            prev_questions
        )

    if q_type == "NEW":
        plans = get_plans(premiums)

        if not plans:
            return (
                f"No plans found for premiums: {premiums}",
                history,
                prev_questions
            )

        answer = generate_answer(question, plans, history)

        new_history_item = {"user": question, "bot": answer, "plans": plans}
        updated_history = history + [new_history_item]
        updated_questions = prev_questions + [question]

        return (answer, updated_history, updated_questions)

    if q_type == "FOLLOW_UP":
        if not history:
            return (
                "Please ask a comparison question first with 2-3 premiums.",
                history,
                prev_questions
            )

        last_plans = history[-1].get("plans", [])

        answer = generate_answer(question, last_plans, history)

        new_history_item = {"user": question, "bot": answer, "plans": last_plans}
        updated_history = history + [new_history_item]
        updated_questions = prev_questions + [question]

        return (answer, updated_history, updated_questions)


# ---------------- Flask Routes ----------------
@app.route('/', methods=['GET'])
def home():
    # Initialize session if not exists
    if 'history' not in session:
        session['history'] = []
    if 'prev_questions' not in session:
        session['prev_questions'] = []
    return render_template('quote_comparison.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.get_json()

        if not data or 'message' not in data:
            return jsonify({"success": False, "error": "message is required"}), 400

        message = data['message'].strip()

        if not message:
            return jsonify({"success": False, "error": "message cannot be empty"}), 400

        # Get session state
        history = session.get('history', [])
        prev_questions = session.get('prev_questions', [])

        # Process the message
        answer, updated_history, updated_questions = chat(message, history, prev_questions)

        # Update session (convert plans to serializable format)
        serializable_history = []
        for item in updated_history:
            serializable_item = {
                'user': item['user'],
                'bot': item['bot']
                # Skip 'plans' as they're not JSON serializable
            }
            serializable_history.append(serializable_item)

        session['history'] = serializable_history
        session['prev_questions'] = updated_questions

        return jsonify({
            "success": True,
            "response": answer
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation history"""
    session['history'] = []
    session['prev_questions'] = []
    return jsonify({"success": True}), 200

# ---------------- Run App ----------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
