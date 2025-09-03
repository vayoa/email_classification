import torch, json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Load category model ----
cat_dir = "outputs/modernbert_category/checkpoint-716"
cat_tok = AutoTokenizer.from_pretrained(cat_dir)
cat_model = (
    AutoModelForSequenceClassification.from_pretrained(cat_dir).to(DEVICE).eval()
)
id2label_cat = json.load(open(f"{cat_dir}/label_maps.json"))["id2label"]

# ---- Load important model ----
imp_dir = "outputs/modernbert_important/checkpoint-716"
imp_tok = AutoTokenizer.from_pretrained(imp_dir)
imp_model = (
    AutoModelForSequenceClassification.from_pretrained(imp_dir).to(DEVICE).eval()
)
id2label_imp = json.load(open(f"{imp_dir}/label_maps.json"))["id2label"]


def predict_category(text, max_length=512):
    enc = cat_tok(text, truncation=True, max_length=max_length, return_tensors="pt").to(
        DEVICE
    )
    with (
        torch.no_grad(),
        torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")),
    ):
        probs = torch.softmax(cat_model(**enc).logits, dim=-1)[0]
    pred = int(torch.argmax(probs))
    return id2label_cat[str(pred)], probs.tolist()


def predict_important(text, threshold=0.5, max_length=512):
    enc = imp_tok(text, truncation=True, max_length=max_length, return_tensors="pt").to(
        DEVICE
    )
    with (
        torch.no_grad(),
        torch.autocast(device_type="cuda", enabled=(DEVICE == "cuda")),
    ):
        probs = torch.softmax(imp_model(**enc).logits, dim=-1)[0]
    p_imp = float(probs[1])
    return ("important" if p_imp >= threshold else "not_important"), p_imp


privacy_notice_email = """Subject: We're Updating our Consumer Terms and Privacy Policy\n\nHello,
We're writing to inform you about important updates to our Consumer Terms and Privacy Policy. These changes will take effect on September 28, 2025, or you can choose to accept the updated terms before this date when you log in to Claude.ai. 
These changes only affect Consumer accounts (Claude Free, Pro, and Max plans). If you use Claude for Work, via the API, or other services under our Commercial Terms or other Agreements, then these changes don't apply to you. 
What's changing?
1. Help improve Claude by allowing us to use your chats and coding sessions to improve our models
    With your permission, we will use your chats and coding sessions to train and improve our AI models. If you accept the updated Consumer Terms before September 28, your preference takes effect immediately. 
    If you choose to allow us to use your data for model training, it helps us:
        Improve our AI models and make Claude more helpful and accurate for everyone
        Develop more robust safeguards to help prevent misuse of Claude
    We will only use chats and coding sessions you initiate or resume after you give permission. You can change your preference anytime in your Privacy Settings.
2. Updates to data retention– your choices and controls
    If you choose to allow us to use your data for model training, we’ll retain this data for 5 years. This enables us to improve Claude through deeper model training as described above, while strengthening our safety systems over time. You retain full control over how we use your data: if you change your training preference, delete individual chats, or delete your account, we'll exclude your data from future model training. Learn more about our data retention practices here.
Learn more and next steps

For detailed information about these changes:
    Read our blog post about these updates
    Review the updated Consumer Terms and Privacy Policy
    Visit our Privacy Center for more information about our practices
    See our Help Center articles on how to manage your privacy settings
    Next time you log into Claude, review the terms and confirm your settings

If you have questions about these updates, please visit our Help Center.
–The Anthropic Team"""


important_email = """Subject: Credit Fraud Detected\n\nHello, this is MAX. We have detected an irregular amount of suspicious activity coming from your credit card, after careful analysis, we have determined your card has possibly been stolen.
If so, it is imperative that you'll act quickly and cancel this card."""

important_email_2 = """Subject: Important: Unusual Activity Detected on Your Card\n\nHello Ron,
We detected unusual activity on your MAX card ending in 3482. For your security, we temporarily placed a hold on certain transactions.
Details of the flagged activity:
Date: September 2, 2025
Amount: $245.87
Location: Online transaction – Merchant not recognized
If you recognize this transaction, no further action is required. If you do not recognize it, please contact us immediately using the phone number on the back of your card or by logging into your account through the official [Bank Name] website or mobile app.
We will never ask you to click on links in an email to verify account details. For your safety, please always navigate to our website by typing the address directly into your browser or using our official mobile app.
Thank you for helping us keep your account secure.
Sincerely,
MAX Fraud Protection Team"""


important_email_3 = """Subject: Action Required: Security Update on Your Work Account\n\nHello Ron,
We have detected a critical vulnerability in the current version of our VPN client. To ensure uninterrupted access to company resources and protect your data, you must install the required update immediately.
What you need to do:
Download the latest VPN client from the company portal.
Install the update before 5:00 PM today.
Restart your device to complete the process.
Failure to update by the deadline will result in automatic suspension of VPN access until the patch is applied.
If you need assistance, please contact the IT Help Desk at 05529349.
Thank you for acting promptly to maintain the security of our systems.
— IT Security Team"""


important_email_4 = """Subject: Action Required: Safety Update on Your Work Account\n\nHello Ron,
We have detected a major vulnerability in the current version of our VPN client. To ensure uninterrupted access to company resources and protect your data, you must install the required update as quickly as possible.
What you need to do:
Download the latest VPN client from the company portal.
Install the update before 5:00 PM today.
Restart your device to complete the process.
Failure to update by the deadline will result in automatic suspension of VPN access until the patch is applied.
If you need assistance, please contact the IT Help Desk at 05529349.
Thank you for acting promptly to maintain the safety of our systems.
— IT Security Team"""


# Example
emails = [
    privacy_notice_email,
    important_email,
    important_email_2,
    important_email_3,
    important_email_4,
]

for email in emails:
    print(email)
    print()
    print("Category:", predict_category(email, max_length=64))
    print("Important:", predict_important(email, threshold=0.5, max_length=64))
    print("-" * 15)
