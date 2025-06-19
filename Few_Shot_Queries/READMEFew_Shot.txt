This folder is organized by how the Few-Shot Queries are laid out.

FS-WIK-WCON-ROLE VS FS-NA-NA-NA 
- The first one is the FULL CONTEXT version of Few-Shot Queries. Meaning it has access to WIK, WCON, and Role. 
- Whereas the other is just a Full-Shot Query, and it just has the query, and enough context to make it be considered Full-Shot.



#1 FS-WIK-WCON-ROLE
#2 FS-WIK-WCON-NA
#3 FS-WIK-NA-ROLE
#4 FS-WIK-NA-NA
#5 FS-NA-WCON-ROLE
#6 FS-NA-WCON-NA
#7 FS-NA-NA-ROLE
#8 FS-NA-NA-NA


For Few-Shot prompting I need to emphasize examples as demonstrations the model should imitate in style and reasoning
Few-shot prompting is about showing how to answer, not limiting the model’s reasoning.


You are analyzing a mathematical workflow composed of multiple tasks.  
Each task processes data and passes it to the next task.  
Tasks have names like I_TO_H and H_TO_G. Data flows through tasks in a directed graph based on dependencies.  
Workflow = workflow.py  

Below are several examples of questions about this workflow, each with optional context:  
- Instance-specific data (WIK)  
- Workflow context (WCON)  
- User or system role (ROLE)  

Each example contains a question and a corresponding answer based only on the information given. Each example is independent and self-contained.  

Study these examples carefully to understand how to answer questions about the workflow using the structure, dependencies, and data transformations.  

Follow the exact format and style shown.  

Now, answer the new question below:  
