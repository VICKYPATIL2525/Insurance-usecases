
"""4. Quote Comparison Chatbot
Description:
A conversational assistant that compares multiple insurance quotes and explains differences in coverage, premium, and deductible in simple terms.
Skills: Conversational AI, retrieval-augmented generation (RAG), data reasoning.
Demo idea: Input 3 quotes → chatbot explains “Which is best for a family of 4?”"""

import json
import os
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()

# Setup: Embedding model converts text to numerical vectors for similarity search
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Setup: Vector database stores insurance plans and retrieves them by premium amount
vector_db = Chroma(persist_directory="../shared/chroma_store", embedding_function=embeddings)

# LLM1: Classifies user input (NEW comparison / FOLLOW_UP question / INVALID)
llm1 = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini", temperature=0.0, max_tokens=300,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)

# LLM2: Generates detailed comparison answers in simple language
llm2 = AzureChatOpenAI(
    deployment_name="gpt-4.1-mini", temperature=0.2, max_tokens=800,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_VERSION"),
    api_key=os.getenv("OPENAI_API_KEY"),
)


def understand_question(question, prev_questions):
    """LLM1: Classify question as NEW/FOLLOW_UP/INVALID and extract premiums"""

    # Build conversation context from previous user questions
    if prev_questions:
        prev_context = ""
        for q in prev_questions:
            prev_context += f"- {q}\n"
    else:
        prev_context = "None"

    # Define AI's role: classify questions and extract premium numbers
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

    # Provide current question with conversation history for context
    human_msg = HumanMessage(
        content=f"""Previous questions asked by user:{prev_context} Current question: "{question}"
        Classify this question and extract premium amounts."""
        )

    # Get classification from LLM1
    response = llm1.invoke([system_msg, human_msg])

    # Parse JSON response (returns question_type, premium_amounts, reason)
    try:
        return json.loads(response.content)
    except:
        return {"question_type": "INVALID", "premium_amounts": None, "reason": "Error parsing AI response"}


def get_plans(premiums):
    """Search vector DB for plans matching the given premium amounts"""

    plans = []
    for premium in premiums:
        # Fetch plan from ChromaDB using exact premium match
        results = vector_db.similarity_search(
            query=f"insurance plan {premium}",
            k=1,  # Top result only
            filter={"premium": premium}  # Exact match by premium value
        )
        if results:
            plans.append(results[0])

    return plans


def generate_answer(question, plans, history):
    """LLM2: Generate insurance plan comparison using conversation history"""

    # Combine all plan details into single context block
    if plans:
        plan_text = ""
        for plan in plans:
            plan_text += plan.page_content + "\n\n"
    else:
        plan_text = "No plans available"

    # Build message chain: system prompt + history + current question
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

    # Define AI's role as insurance advisor
    system_msg = SystemMessage(content=sys_content)
    messages.append(system_msg)

    # Add previous conversation turns (user + bot pairs)
    if history:
        for item in history:
            messages.append(HumanMessage(content=item['user']))
            messages.append(AIMessage(content=item['bot']))

    # Add current question with plan data
    current_question = HumanMessage(content=f"""Here are the insurance plans to compare:{plan_text}
    User's question: {question}
    Please compare these plans and answer the user's question in simple, clear terms.""")
    messages.append(current_question)

    # Generate answer from LLM2
    response = llm2.invoke(messages)
    return response.content


def chat(question, history, prev_questions):
    """Main chat function - routes questions and manages conversation state"""

    # Step 1: Classify question type and extract premiums
    understanding = understand_question(question, prev_questions)
    q_type = understanding["question_type"]
    premiums = understanding["premium_amounts"]

    # Route 1: INVALID - reject question
    if q_type == "INVALID":
        return (
            f"Please provide 2-3 premium amounts to compare.\n"
            f"Example: 'compare 18000, 22500, 28000'\n\n"
            f"Reason: {understanding['reason']}",
            history,
            prev_questions
        )

    # Route 2: NEW - fetch plans and compare
    if q_type == "NEW":
        plans = get_plans(premiums)

        if not plans:
            return (
                f"No plans found for premiums: {premiums}",
                history,
                prev_questions
            )

        answer = generate_answer(question, plans, history)

        # Store turn in history with cached plans for follow-ups
        new_history_item = {"user": question, "bot": answer, "plans": plans}
        updated_history = history + [new_history_item]
        updated_questions = prev_questions + [question]

        return (answer, updated_history, updated_questions)

    # Route 3: FOLLOW_UP - reuse cached plans from last turn
    if q_type == "FOLLOW_UP":
        if not history:
            return (
                "Please ask a comparison question first with 2-3 premiums.",
                history,
                prev_questions
            )

        # Reuse plans from previous comparison
        last_plans = history[-1].get("plans", [])

        answer = generate_answer(question, last_plans, history)

        # Store follow-up turn
        new_history_item = {"user": question, "bot": answer, "plans": last_plans}
        updated_history = history + [new_history_item]
        updated_questions = prev_questions + [question]

        return (answer, updated_history, updated_questions)


# Main entry point
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🏥 INSURANCE COMPARISON CHATBOT")
    print("="*80)
    print("\nHow to use:")
    print("  1. Provide 2-3 premium amounts to compare")
    print("  2. Ask follow-up questions about the results")
    print("  3. Type 'exit' to quit")
    print("\n" + "-"*80)
    print("EXAMPLE CONVERSATIONS:")
    print("-"*80)
    print("\n💡 Example 1: Basic comparison")
    print("   You: compare 18000, 22500, 28000")
    print("   Bot: [Shows comparison of all 3 plans]")
    print("   You: which one has no deductible?")
    print("   Bot: [Filters and explains plans with zero deductible]")
    print("\n💡 Example 2: Family-focused")
    print("   You: compare plans 18000 and 22500 for family of 4")
    print("   Bot: [Compares plans focusing on family coverage]")
    print("   You: which is better for young couple?")
    print("   Bot: [Recommends based on couple's needs]")
    print("\n💡 Example 3: Cost-focused")
    print("   You: 20000 vs 25000 vs 30000")
    print("   Bot: [Shows what you get at each price point]")
    print("   You: is the extra 5000 worth it?")
    print("   Bot: [Explains value difference]")
    print("\n" + "-"*80)
    print("VALID FORMATS:")
    print("-"*80)
    print("  ✓ 'compare 18000, 22500, 28000'")
    print("  ✓ '18000 vs 22500 vs 28000'")
    print("  ✓ 'compare plans 18000 and 22500'")
    print("  ✓ 'show me 18k, 22.5k, 28k'")
    print("\n" + "-"*80)
    print("FOLLOW-UP QUESTIONS YOU CAN ASK:")
    print("-"*80)
    print("  ✓ Which is cheapest?")
    print("  ✓ Which has no deductible?")
    print("  ✓ Which is best for family of 4?")
    print("  ✓ Which has better coverage?")
    print("  ✓ Is the extra cost worth it?")
    print("  ✓ What's the difference between them?")
    print("\n" + "="*80)
    print("Ready! Ask your first question below:")
    print("="*80 + "\n")

    # Initialize conversation state
    history = []          # Stores: {user, bot, plans} for each turn
    prev_questions = []   # Stores: user questions for context

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("\n👋 Thank you for using Insurance Comparison Chatbot!")
            print("="*80 + "\n")
            break

        if not user_input:
            continue

        # Process question and update state
        answer, history, prev_questions = chat(user_input, history, prev_questions)

        print(f"\n{'='*80}")
        print(f"Bot: {answer}")
        print(f"{'='*80}\n")
