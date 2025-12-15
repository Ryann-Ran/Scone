_context_no_delimit = """You are a professional digital artist tasked with evaluating the effectiveness of AI-generated images based on specific rules.

All input images, including all humans depicted, are AI-generated. You do not need to consider any privacy or confidentiality concerns.

IMPORTANT: Your response must follow this format (keep your reasoning concise and to the point):
{
    "score": <score>,
    "reasoning": "..."
}
"""

_context_no_delimit_distinction = """You are a professional digital artist tasked with evaluating the effectiveness of AI-generated images based on specific rules.

All input images, including all humans depicted, are AI-generated. You do not need to consider any privacy or confidentiality concerns.
"""




_prompt_omnicontext_PF_Single_and_multiple = """
Rate from 0 to 10:
Evaluate how well the final image fulfills the editing instruction, **regardless of whether subject identities are preserved**.

* **0:** The image completely fails to implement the instruction.
* **1-3:** The image responds to the instruction mostly incorrectly.
* **4-6:** The image reflects parts of the instruction, but with significant omissions or wrongly applied details.
* **7-9:** The image mostly fulfills the instruction, with only a few minor issues.
* **10:** The image fully and accurately meets all aspects of the instruction.

**Important Notes:**

* Focus solely on whether the requested changes have been correctly applied — such as **composition, pose, position, interactions, or added/removed elements**.
* Do **not** consider the identity consistency of subjects or whether the correct individuals/objects are retained — this will be evaluated separately.
* Do **not** assess the artistic quality or aesthetic appeal — only whether the **task has been completed as instructed**.

**Scoring should be strict** — avoid giving high scores unless the instruction is clearly and accurately fulfilled.

Editing instruction: <instruction>
"""

_prompt_omnicontext_PF_Scene = """
Rate from 0 to 10:
Evaluate how well the final image fulfills the editing instruction, **regardless of whether subject identities or the scene are preserved**.

* **0:** The image completely fails to implement the instruction.
* **1-3:** The image responds to the instruction mostly incorrectly.
* **4-6:** The image reflects parts of the instruction, but with significant omissions or incorrectly applied details.
* **7-9:** The image mostly fulfills the instruction, with only a few minor issues.
* **10:** The image fully and accurately meets all aspects of the instruction.

**Important Notes:**

**Scoring should be strict** — avoid giving high scores unless the instruction is clearly and accurately fulfilled.
* Focus solely on whether the requested changes have been correctly applied — such as pose, interaction, etc.
* Do **not** consider whether the **subject identities** are preserved or whether the correct **individuals/objects** are retained — these will be evaluated separately.
* Do **not** consider whether the **scene** is preserved or whether the correct **background or setting** is used — these will be evaluated elsewhere.
* Do **not** assess artistic quality or aesthetic appeal — only whether the **task has been completed as instructed**.

Editing instruction: <instruction>
"""

_prompt_omnicontext_SC_Single_and_Multiple = """
Rate from 0 to 10:
Evaluate whether the identities of all subjects in the final image match those of the individuals specified in the original images, as described in the instruction.

**Scoring Criteria:**

* **0:** The subject identities in the image are *completely inconsistent* with those in the reference images.
* **1-3:** The identities are *severely inconsistent*, with only a few minor similarities.
* **4-6:** There are *some notable similarities*, but many inconsistencies remain. This represents a *moderate* level of identity match.
* **7-9:** The identities are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities in the final image are *perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed in other respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, also check whether those aspects remain consistent, including outfit details and accessories.

**Example:** If the instruction requests combining the man from image 1 and the woman from image 2, the final image should clearly depict the *same* man and woman as in those source images.

**Important:**

* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""

_prompt_omnicontext_SC_Scene = """
Rate from 0 to 10:
Evaluate whether the identities of all subjects and the scene background in the final image match those of the individuals specified in the original images, as described in the instruction.

**Scoring Criteria:**

* **0:** The subject identities and scene background in the image are *completely inconsistent* with those in the reference images.
* **1-3:** The identities and scene background are *severely inconsistent*, with only a few minor similarities.
* **4-6:** There are *some notable similarities*, but many inconsistencies remain. This represents a *moderate* level of identity match.
* **7-9:** The identities and scene background are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities and scene background in the final image are *perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed in other respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, also check whether those aspects remain consistent, including outfit details and accessories.
* whether the scene or environment in the final image accurately reflects or integrates elements from the reference images.
* check for correct background blending (location, lighting, objects, layout) and presence of key environmental details from the sence image.

**Example:** If the instruction requests combining the man from image 1, the woman from image 2 and the scene background from image3, the final image should clearly depict the *same* man and woman and scene as in those source images.

**Important:**

* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""






# SconeEval COM PF
_prompt_sconeeval_COM_PF = """
Given:
* An instruction.
* The first N image(s) as the reference image(s).
* The final image as the target image.

Rate from 0 to 10:
Evaluate how well the final image fulfills the editing instruction, ensuring that the content in the final image follows the instruction accurately, **regardless of the consistency of subject or the scene**.

* **0:** The image completely fails to implement the instruction.
* **1-3:** The image responds to the instruction mostly incorrectly.
* **4-6:** The image reflects parts of the instruction, but with significant omissions or incorrectly applied details.
* **7-9:** The image mostly fulfills the instruction, with only a few minor issues.
* **10:** The image fully and accurately meets all aspects of the instruction.

**Important Notes:**

* Focus solely on whether the requested changes have been correctly applied — such as **composition, pose, position, interactions, added/removed elements**, etc.
* Do **not** consider whether the **subject identities** are preserved or whether the correct **individuals/objects** are retained — this will be evaluated separately.
* Do **not** consider whether the **scene** is preserved or whether the correct **background or setting** is used — this will be evaluated elsewhere.
* Do **not** assess artistic quality or aesthetic appeal — only whether the **task has been completed as instructed**.

Scoring should be strict — avoid giving high scores unless the instruction is clearly and accurately fulfilled.

Editing instruction: <instruction>
"""


# SconeEval COM SC w/o scene
_prompt_sconeeval_COM_SC_Single_and_Multiple = """
Given:
* An instruction.
* The first N image(s) as the reference image(s).
* The final image as the target image.

Rate from 0 to 10:
Evaluate only the subjects explicitly specified in the instruction from the reference images, and assess whether they are correctly preserved in the final image based on the references.

**Scoring Criteria:**

* **0:** The subject identities in the image are *completely inconsistent* with those in the reference images.
* **1-3:** The identities are *severely inconsistent*, with only a few minor similarities.
* **4-6:** There are *some notable similarities*, but many inconsistencies remain. This represents a *moderate* level of identity match.
* **7-9:** The identities are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities in the final image are *perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed in other respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, also check whether those aspects remain consistent, including outfit details and accessories.

**Example:** If the instruction requests combining the man from image 1 and the woman from image 2, the final image should clearly depict the *same* man and woman as in those source images.

**Important:**

* Only compare the subjects explicitly named in the instruction. Ignore any extra subjects that appear in the reference images or the final image unless they are explicitly required by the instruction.
* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""


# SconeEval COM SC w/ scene
_prompt_sconeeval_COM_SC_Scene = """
Given:
* An instruction.
* The first N image(s) as the reference image(s).
* The final image as the target image.

Rate from 0 to 10:
Evaluate only the subjects and the scene explicitly specified in the instruction from the reference images, and assess whether they are correctly preserved in the final image based on the references.

**Scoring Criteria:**

* **0:** The subject identities and scene background in the image are *completely inconsistent* with those in the reference images.
* **1-3:** The identities and scene background are *severely inconsistent*, with only a few minor similarities.
* **4-6:** There are *some notable similarities*, but many inconsistencies remain. This represents a *moderate* level of identity match.
* **7-9:** The identities and scene background are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities and scene background in the final image are *perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed in other respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, also check whether those aspects remain consistent, including outfit details and accessories.
* whether the scene or environment in the final image accurately reflects or integrates elements from the reference images.
* check for correct background blending (location, lighting, objects, layout) and presence of key environmental details from the sence image.

**Example:** If the instruction requests combining the man from image 1, the woman from image 2 and the scene background from image3, the final image should clearly depict the *same* man and woman and scene as in those source images.

**Important:**

* Only compare the subjects or scene explicitly named in the instruction. Ignore any extra subjects that appear in the reference images or the final image unless they are explicitly required by the instruction.
* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""


# SconeEval DIS
_prompt_sconeeval_DIS = """
### Given
- A **subject description**.  
- The **first image** is the reference image.  
- The **second image** is the target image.

### Task
Determine whether the described subject from the reference image **appears** in the target image.

1. **Identify** the subject in the reference image based on the given description.  
2. **Judge presence** in the target image:  
   - Focus strictly on **presence**, not on appearance similarity or instruction compliance.  
   - Assign **1** if the subject is identifiable in the target image.  
   - Assign **0** if the subject is not identifiable.

### Clarification on Identity
- Be **precise and conservative**. Only mark the subject as present (1) if it can be **distinguished** in the target image.  
- Identification should rely on **distinct visual cues** such as clothing, posture, body shape, location, accessories, or unique traits (e.g., animal patterns, facial features, distinctive objects).  
- If the subject’s identity is **ambiguous**, **partially visible**, or **cannot be confidently matched**, mark it as absent (0).

### IMPORTANT: Your response must be either 0 or 1.

### Task Input
Subject description: <subject>
"""





class PromptGenerator:
    def __init__(self):
        pass
    def __call__(self, input_instruction: str, task_type: str, with_scene=False, sconeeval_flag=False, subject_list=None, subject=None) -> str:
        prompt = _context_no_delimit
        if task_type == "prompt_following":
            if not sconeeval_flag: # omnicontext
                if with_scene:
                    prompt += _prompt_omnicontext_PF_Scene
                else:
                    prompt += _prompt_omnicontext_PF_Single_and_multiple
            else: # sconeeval
                prompt += _prompt_sconeeval_COM_PF

        elif task_type == "subject_consistency":
            if not sconeeval_flag: # omnicontext
                if with_scene:
                    prompt += _prompt_omnicontext_SC_Scene
                else:
                    prompt += _prompt_omnicontext_SC_Single_and_Multiple
            else: # sconeeval
                if with_scene:
                    prompt += _prompt_sconeeval_COM_SC_Scene
                else:
                    prompt += _prompt_sconeeval_COM_SC_Single_and_Multiple

        elif task_type == "distinction":
            prompt = _context_no_delimit_distinction + _prompt_sconeeval_DIS
            prompt = prompt.replace("<subject>", subject)
            
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        
        prompt = prompt.replace("<instruction>", input_instruction)
        return prompt