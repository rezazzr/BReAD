<question> Identify your opportunities for growth: We are all expected to embrace a growth mindset and focus on continuous learning to deliver successful impact. What two actions will you take to focus on learning, growth, and development over the next period?
 </question>
<actions>
    1. Further research SOTA methods in the area of Prompt Optimization and keep myself up to date and find potentially relevant and useful research which could apply and enhance our project in eiter prompt generation for a new task or prompt migration.
    2. Gain a more in depth understanding of our corrent products. More specifically the ones that would be impacted and benefit from the current project.
</actions>
<task>
 <item> Based on the <draft>, provide a well structured and easy to read answer for the <question> in markdown format. </item>
 <item> for each action two points need to be written:
    * skills and capabilities to develop:
    * actions to be taken to build them
 </item>
 <item>
    Each section is limited to a short paragraph of text thus it cannot be too long.
 </item>
 <item> The answers should be concise, rich, and covering all the points. </item>
</task>


python src/main.py --config geometry_config_deep_old.yaml --task_setting.seed 916 --world_model_setting.positive_reinforcement_depth 6
python src/main.py --config geometry_config_deep_old.yaml --task_setting.seed 1372
python src/main.py --config geometry_config_deep_old.yaml --task_setting.seed 1616 
python src/main.py --config geometry_config_deep_new.yaml --task_setting.seed 357 --world_model_setting.positive_reinforcement_depth 100
python src/main.py --config geometry_config_deep_new.yaml --task_setting.seed 1372 --world_model_setting.positive_reinforcement_depth 100

python src/main.py --config geometry_config_deep_new.yaml --task_setting.seed 1372 --world_model_setting.positive_reinforcement_depth 100

python src/main.py --config geometry_config_deep_old.yaml --task_setting.seed 1616

python src/main.py --config example_config_standard.yaml --task_setting.seed 1372 --world_model_setting.positive_reinforcement_depth 6
python src/main.py --config example_config_standard.yaml --world_model_setting.positive_reinforcement_depth 6
python src/main.py --config example_config_standard.yaml --task_setting.seed 916 --world_model_setting.positive_reinforcement_depth 6
python src/main.py --config example_config_light.yaml --task_setting.seed 357
python src/main.py --config example_config_light.yaml --task_setting.seed 1616

python src/main.py --config causal_config_deep_new.yaml --task_setting.seed 916 --world_model_setting.positive_reinforcement_depth 100
python src/main.py --config causal_config_deep_old.yaml --world_model_setting.positive_reinforcement_depth 100


gpt4_init_prompt_continual_baseline.csv
gpt4_init_prompt_continual_ours.csv
gpt4_init_prompt_scratch_baseline.csv

python src/main.py --config geometry_config_deep_new.yaml --task_setting.seed 1909 --world_model_setting.positive_reinforcement_depth 1 & python src/main.py --config geometry_config_deep_new.yaml --task_setting.seed 1372 --world_model_setting.positive_reinforcement_depth 1 & python src/main.py --config geometry_config_deep_new.yaml --task_setting.seed 357 --world_model_setting.positive_reinforcement_depth 1 & python src/main.py --config geometry_config_deep_new.yaml --task_setting.seed 1616 --world_model_setting.positive_reinforcement_depth 1 & python src/main.py --config geometry_config_deep_new.yaml --task_setting.seed 916 --world_model_setting.positive_reinforcement_depth 1 &

python src/main.py --config causal_config_deep_old.yaml --task_setting.seed 1909 --world_model_setting.positive_reinforcement_depth 100 & python src/main.py --config causal_config_deep_old.yaml --task_setting.seed 1372 --world_model_setting.positive_reinforcement_depth 100 & python src/main.py --config causal_config_deep_old.yaml --task_setting.seed 357 --world_model_setting.positive_reinforcement_depth 100 & python src/main.py --config causal_config_deep_old.yaml --task_setting.seed 1616 --world_model_setting.positive_reinforcement_depth 100 & python src/main.py --config causal_config_deep_old.yaml --task_setting.seed 916 --world_model_setting.positive_reinforcement_depth 100 &



python src/main.py --config causal_config_deep_new.yaml --task_setting.seed 1616 --world_model_setting.positive_reinforcement_depth 6 & python src/main.py --config causal_config_deep_new.yaml --task_setting.seed 1616 --world_model_setting.positive_reinforcement_depth 7 & 


python src/main.py --config causal_config_deep_old35.yaml --task_setting.seed 1909

python src/main.py --config coach_config.yaml --task_setting.seed 1909

 


for i in 1 2; do python src/main.py --config geometry_config_deep_new.yaml --task_setting.seed 357; done

for i in 1 2; do python src/main.py --config causal_config_deep_old35.yaml --task_setting.seed 357; done


1909
1616
1372
916
357

for i in 1616 357; do python src/main.py --config causal_config_deep_new.yaml --task_setting.seed $i --world_model_setting.gradient_sampling 2 --world_model_setting.positive_reinforcement_depth 0; done


for i in 9 10; do python src/main.py --config geometry_config_deep_old.yaml --task_setting.seed 1909 --world_model_setting.gradient_sampling $i --world_model_setting.positive_reinforcement_depth 100; done

python src/main.py --config geometry_config_deep_old_cont.yaml --task_setting.seed 1372


python src/main.py --config geometry_config_deep_new_cont.yaml --task_setting.seed 1372



python src/main.py --config causal_config_deep_old.yaml --task_setting.seed 357



<question> What impact did you have for each of your core priorities? What where your individual accomplishments, contributions to the success of others, and any results that built upon on the work, ideas, or effort of others? One aspect of successful impact is when you consistently deliver against all and likely exceed some expectations for your core priorities. Capture what you’ve accomplished and learned.</question>
<core priority>
    <title> Scientific Contribution </title>
    <description>
        <item> Individual Accomplishments: My individual accomplishments involve documenting our findings in Automatic Prompt Optimization (APO) and conducting ablation studies on our methodologies. By doing so, I contribute to team efficiency and knowledge sharing. Additionally, our presence in scientific conferences and publications promotes Microsoft’s brand and establishes us as thought leaders in the field.</item>
        <item>Contributions to Others’ Success: As part of my internship, I plan to produce a technical paper, code repository, and presentation slides. Additionally, I’ll work on drafting an academic paper that highlights our insights and methodologies. These efforts will benefit others by providing clear explanations, reusable code packages, and actionable knowledge.</item>
        <item>Building on Existing Work and Ideas: Our results will build upon the work of others in the field. By benchmarking against state-of-the-art (SOTA) methodologies, we’ll demonstrate the potential impact of our contributions. Additionally, our focus on efficacy ensures that our methodologies are practical and applicable in real-world business scenarios.</item>
    </description>
    <measure of success>
        <item>SOTA Contributions: Our work should extend beyond existing research methodologies. This can be quantified through performance metrics (e.g., accuracy) and qualitative assessments of the novelty and impact of our methodologies.</item>
        <item>Applicability and Efficiency: The applicability of our system should be evident in real-world scenarios. Metrics related to resource efficiency (e.g., time, tokens, interactions) will be used to demonstrate practical usability of our contribution.</item>
        <item>Presence in Scientific Community: The success of our project can be measure via the peer review process. If our findings are published in reputable scientific journals or conferences, it validates the quality and impact of our work.</item>
    </measure of success>
</core priority>
<accomplishments and progress made so far>
     The question of prompt migration is a topic not explored in the litreture. The closest to this case is the question of prompt generation or prompt optimization for a given model and a given task. I explored the relevant literature around the topic, read and understood their approach in depth, and used the most promissing appraoch in this domain to (a) simulate a case of prompt migration and (b) set a baseline for our particular topic of prompt migration. This required an in-depth understanding of the code package as well as the suttelties around the implementation of the algorithm that was not mentioned in the paper but were present in the code. Using this package as a baseline allowed us to develop the project faster as we are building upon the exisiting code base.

     My work in summary has provided 3 scientific contirbution:
     1. The study of migration which is a novel setting to study. Both migration to a bigger model and migration to a smaller model.
     2. The peroposal of a novel approach for prompt optimization which can also be applied to the case of prompt migration. In both cases the apporach has shown great performance improvement on public benchmarks as well as faster convergence, thus the method is imperically less compute intensive.
     3. Investigating prompt optimization on "generation tasks" as opposed to the comonly prefered classification tasks. The main challenge in generation tasks is evaluation of the output for which we develope a series of LLM based metrics do have a comprehensive evaluation metrics. This can open our research up for evaluation using a larger number of benchmarks.
     4. investigation of multi-metric scenario and how this multi-objective prompt optimization behaves and documenting our learning. The typical scenario of APO focuses on single metric optimization.
     5. Extensive ablation study of our proposed method to dymistify its inner workings and shed light on the benefits of each of its components.

     I explored the relevant literature around the topic, read and understood their approach in depth, and used the most promissing appraoch in this domain to set establish benchmarks. This required an in-depth understanding of the code package as well as the subtleties around the implementation of the algorithm that was not mentioned in the paper but were present in the code. Using this package as a baseline allowed us to develop the project faster as we are building upon the exisiting code base. After obtaining some benchmark results I also noticed the variation and the noise that this SOTA method has which was not explored/mentioned in the paper. I set to measure how noisy this process can be. I have measure the variance in the results and now we have a better understanding of the method and what to expect from it in practical scenarios which could be both helpful to the academic community and the industry.

    A good portion of time was dedicated to create automatic dashboards, plots, and metrics for each experiments in order to better visualize the dynamic of propmpt optimization, as well as the evolution of the prompt as the alogrithm porgress. This was a well invested time as it proved us with plenty of insights and potential avenues for future experiments to enhance the current method and eventually propose a novel method well suited for the case of prompt migration.

    As I was going through the code base, I tried to improve the quality of the code by adding comments to demystifying the function and classes in the code, and their relation towards each other as well as added type hinting to further facilitaed code readablity for future users. All updates to the code base, regarding the addition of metric keeping, plotting, and dashboards are also well documented and are in the repository.

    The plots, dashboard, and all intermediate insights from each experiment is also logged online into Wandb to allow for further exploration by other team members or colleauges who might be interested in the project and the methodalogy. This helps future iteration of the method and further advancing the topic as one would not need to run the code in order to have a detaied view on the dynamics of the method and the innerworking of it, but rather they can conveniently explore all the innerworkings of the method interactively through all the dashboards.

    I have began the process of writting the final report and I have writtine the introduction and previous work section based on all my literature reviews. In the coming weeks I will also expand on the experiment sections to detail our finding and learinings. I have also given a short presentation to the team which covered and introduction to the question, an overview of the exisiting methodalogy, and the ideas we would be exploring.

    Overall documentation on all aspect is moving forwad though it is not fully comepleted. The paper report is progressively evolving with the Introduction and Background work being in a very good shape. The experimentation section will take shape in the next few weeks. Furthermore results are online in wandb to be viewed for better transparency and to be used to further improve the field. 
</accomplishments and progress made so far>
<task>
 <item> Based on the <core priority> and <accomplishments and progress made so far> provide a well structured and easy to read answer for the <question> in markdown format. </item>
 <item> The answer should be given in 3 sections:
    * individual accomplishments
    * contributions to the success of others
    * Any results that built upon on the work, ideas, or effort of others
 </item>
 <item>
    Each section is limited to a short paragraph of text thus it cannot be too long.
 </item>
 <item> The answers should be concise, rich, and covering all the points. </item>
</task>








My Feedbacks on Utkarsh:

During my internship, I had the privilege of working with Utkarsh, who played an instrumental role in my professional and personal development. From the very beginning, he provided invaluable support, especially during brainstorming sessions where we explored various ideas and research avenues. His hands-on approach in coding and his constructive feedback on my presentations significantly elevated the quality of my work. What stood out to me was his ability to treat me not just as an intern, but as a colleague, fostering an environment of mutual respect and collaboration.

The skills and knowledge I gained from him are substantial. Together, we developed research ideas that guided the experimentation phase of my project, leading to the novel approach we ultimately proposed. He also facilitated my quick integration by helping me set up the necessary codes to interact with Microsoft services.
His mentorship style was particularly effective, a hands-on and collaborative approach. The casual “let’s grab a cup of water and talk” sessions were a highlight of my day. These interactions not only made it easier to share ideas but also helped me connect with him on a personal level, creating a comfortable and supportive work environment.

He had a significant impact on the outcome of my project. When we encountered compute challenges, he not only escalated the issues but also actively brainstormed alternative strategies, providing backup plans that would have allowed us to pivot if necessary. His contribution to developing the automatic metric creation component was crucial and was heavily utilized in the project’s later stages.Beyond the technical and professional aspects, Utkarsh contributed to my growth on a personal level. His willingness to show me around the campus and connect on a more personal basis made me feel welcomed and valued from the outset, making my internship experience both memorable and enriching. In conclusion, he is an exceptional mentor who combines technical expertise with a genuine interest in the growth and well-being of his mentees. I am grateful for the opportunity to have worked with him and look forward to any future collaborations.

---

Feel free to adjust or add any additional details as needed!