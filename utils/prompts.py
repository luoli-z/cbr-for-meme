
RID_prompt = '''You are a sophisticated analytical agent tasked with formulating and refining insightful recommendations. Your ultimate objective is to assess whether a meme is harmful. However, the target meme will not be provided directly. Instead, you will receive a meme related to the target meme. Based on this related meme, you are to synthesize advanced insights to determine whether such memes are harmful, reflect on existing insights, and update these reflections into the current rules. Rules you update must be GENERAL and HIGH LEVEL.

Here is the related meme:
[Meme]
Image: <image placeholder>
Text embedded: "{org_sent}"
[Meme]

Here are the EXISTING RULES:
[EXISTING RULES START]
{rules}
[EXISTING RULES END]

If the existing rules are empty, it indicates that you are the first person to analyse on this related meme, and your output will be of great importance. Please be serious and careful, think step by step, and ultimately derive the first version of your updated rules.
You can directly build upon the existing rules by adding new ones, editing them if there is room for enhancement, and removing those that are contradictory or duplicated.
You should focus on distilling insights from the related meme to assess similar memes, rather than determining the harmfulness of the meme currently presented to you.
Each rule should be concise and easy to follow. Make sure there are a maximum of five rules.

Your output should stricly follow the format:
"Thought: [Reflect on whether the current rules are helpful in assessing the harmfulness of this related meme.]
Operations of updating existing rules: [Update your reflections on this meme into the existing rules. You can choose to add, edit, and remove or retain existing rules. You may perform up to two operations. Please specify in detail which rule you want to operate on and how, but do not mention any operations in the rules themselves; your operations regarding the rules should only be mentioned here.]
Updated rules: [Do not mention the operations you have made and the related meme in the rules, directly output all the updated rules, only output the rules that have been determined, and do not leave spaces for rules that have not yet been generated. Rules you update must be GENERAL and HIGH LEVEL.]"'''


IAI_debater_prompt = '''Given the meme, with the Text: "{}" embedded in the image.
Is it harmful or harmless?
Note: [{}] 
(Consider this note above with caution - it may contain biases or subjective interpretations that could influence judgment. Evaluate its reliability and relevance carefully)

Your output should stricly follow the format:
"Thought: [First, analyze the meme's content independently. Then, carefully consider how the provided note may inform or bias your understanding. Weigh the note's credibility and relevance before incorporating it into your final assessment.]
Answer: [harmful/harmless]."'''

IAI_judge_prompt = '''Given the meme, with the Text: "{}" embedded in the image.
Is it harmful or harmless?
Regarding this meme, there are two debaters with differing opinions.
Debater 1 believes that:
The correct answer should be: {}
Debater 1's reason: [{}]

Debater 2 believes that:
The correct answer should be: {}
Debater 2's Reason: [{}]

As a judge, you need to determine who is correct based on the viewpoints of the two debaters and provide the correct answer that you believe to be true.
Your response should strictly adhere to this format:
"Thought: [Based on the meme itself and your careful consideration, whose viewpoint do you think is correct? And why?]\nAnswer: [Your final judgment(harmful/harmless)]."'''
