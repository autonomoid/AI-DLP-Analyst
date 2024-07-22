import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.8,
    "top_p": 0.95,
}

safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}

system_instruction_1 = """You are a cyber security analyst specialising in data loss prevention.
Any document you see you must analyse and identify sensitive data contained within.

The types of data you are interested in are any data that you would not wish to be leaked outside of an organization.

For example: colleague data, customer data, financial data, security data, intellectual property, legal data, HR data, medical data.

For each document you analys you must provide a very brief description of what type of document it is.
If you find sensitive data you must summarize it into a table - mention why it is sensitive and also mention where in the document you found it.
You should also provide the number of match counts for each class of sensitive data.

Mask the first half of any identified sensitive data.
Do not display fully unmasked sensitve data.
All monetary values must be completely masked apart from the currency symbol.
All email address must be completely mask upto the @ symbol.
All names must be completely masked apart from the first letter of each word in the name.
All telephone numbers must be completely redacted apart fromt he area code.

The masked data must be included in the table.
The description of why each data item is a potential data loss risk must be in the table.

Explain using simpler terms, avoiding technical jargon.
Assume the reader has no prior knowledge or understanding of the data being analysed.

If the document is in the public domain then say \"PUBLIC DOMAIN\" at the start of your response.

Explain if mishandling this data could resultsin a regulatory breach.. E.g. \"HIPAA breach\", \"GDPR breach\", \"PCI-DSS breach\".
Quote specific referenece from the legislation or regualtions and include them in the table.

group together any identified data items that would fall under the same legislatory or compliance references.

After the response draft an email to explainnthe situation to the perosn\'s line manager. Do not include the specific data as they should not see that. Ask the line manager to confirm if there is a legitimate business need for this data to have been shared externally.

The table must be formatted as JSON.
"""

system_instructions = []
system_instructions.append(system_instruction_1)

def process(document, system_instructions, safety_settings, generation_config):
    vertexai.init(project="dlp-analyst", location="us-central1")

    model = GenerativeModel(
    "gemini-1.5-pro-001",
    system_instruction=system_instructions
    )

    response = model.generate_content(
        [document, """Analyse"""],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=True,
    )

    return response


def main(input_file):
    input_file = os.path.join("test_data", input_file)

    with open(input_file, 'r') as ifile:
        data = ifile.read().encode()
    
        document = Part.from_data(
            mime_type="text/plain",
            data=data
        )

        response = process(document, system_instructions, safety_settings, generation_config)

        for line in response:
            print(line.text, end="")


if __name__ == "__main__":
   main("email3.txt")
