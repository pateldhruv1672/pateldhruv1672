<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=rect&height=120&color=0:0f172a,100:1e3a5f&text=Dhruv%20Patel&fontColor=ffffff&fontSize=48&fontAlignY=42&desc=Robotics%20%26%20Applied%20AI%20Research%20Engineer&descAlignY=74&descSize=17" alt="Dhruv Patel" />
</p>

<p align="center">
  <a href="https://pateldhruv1672.github.io/"><img src="https://img.shields.io/badge/Portfolio-pateldhruv1672.github.io-0f172a?style=flat-square" alt="Portfolio"/></a>
  <a href="https://www.linkedin.com/in/dhruvpatel16/"><img src="https://img.shields.io/badge/LinkedIn-dhruvpatel16-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
  <a href="https://huggingface.co/dhruv1672"><img src="https://img.shields.io/badge/Hugging%20Face-dhruv1672-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"/></a>
  <a href="mailto:dhruvkumarkamleshbhai.patel@sjsu.edu"><img src="https://img.shields.io/badge/Email-SJSU-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"/></a>
  <img src="https://komarev.com/ghpvc/?username=pateldhruv1672&style=flat-square&color=0f172a&label=profile+views" alt="Profile views"/>
</p>

<p align="center">
  <a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=16&duration=3200&pause=900&color=1E3A5F&center=true&vCenter=true&width=720&lines=Robot+learning+in+simulation+%E2%86%92+hardware;Multi-robot+agents+with+shared+memory;Evidence+first%3A+every+number+is+measured" alt="Typing SVG" /></a>
</p>

I am a graduate researcher in the **Robotics & Digital Twin Lab at San José State University** (M.S. Applied Data Science). My work sits where robot learning, simulation and agentic AI meet: building the environments, datasets and policies that let robots do useful multi-step tasks, and measuring honestly how well they do them. Before graduate school I shipped production retrieval and ML systems at **Bain & Company** and distributed data services at **Samsung Research**.

```yaml
whoami:
  role:      Graduate Researcher, Robotics & Digital Twin Lab @ SJSU
  focus:     [robot learning, VLA fine-tuning, multi-robot agents, sim-to-real evaluation]
  sim:       [Isaac Sim 6, Isaac Lab 3, MuJoCo, Gazebo]
  robots:    [Unitree Go2, UFactory xArm 7, Unitree G1 (sim), SO-101 (sim)]
  ml:        [PyTorch, LeRobot, SmolVLA, diffusers + PEFT, vLLM, NVFP4]
  agents:    [LangGraph, MCP, Neo4j memory, RAG]
  compute:   [NVIDIA DGX Spark (GB10), RTX 5080 laptop]
  previously: [Bain & Company (SWE II), Samsung R&D Institute India (SWE)]
  principle: "report what was measured, including the negative results"
```

---

## Recent work (Sep to Nov 2026)

### Cross-Bot: a multi-robot service system &nbsp;·&nbsp; M.S. thesis, in progress

A bimanual xArm 7 coffee station, a Unitree G1 that carries trays between tables and a Go2 that explores and navigates. Each robot runs its own LangGraph agent; they coordinate through a shared Neo4j memory and durable handover records. Everything runs in Isaac Sim today, designed so the same stack moves to the real robots. The code stays private while the thesis is in progress; the datasets are public.

<p align="center">
  <img src="assets/g1-tray-transfer.jpg" width="92%" alt="Unitree G1 moving a tray of cups from a serving table to the centre of a dining table, eight frames"/>
  <br/><sub>Unitree G1 (29 DoF, BrainCo Revo2 hands) transferring a tray between two tables, one recorded episode (seed 2301).</sub>
</p>

<p align="center">
  <img src="assets/xarm-coffee-serving.jpg" width="92%" alt="Bird's-eye view of two xArm 7 arms placing a cup under a coffee machine and moving it toward a tray"/>
  <br/><sub>Bimanual xArm 7 coffee serving, bird's-eye camera from a recorded run: cup to the outlet, brew, handover to the tray arm.</sub>
</p>

| Output | Contents | Notes |
|---|---|---|
| [`crossbot-g1-tray-delivery-rgbd`](https://huggingface.co/datasets/dhruv1672/crossbot-g1-tray-delivery-rgbd) | 50 episodes, 64,741 frames at 10 Hz, head D435i RGB + lossless depth, 51-D state / 44-D action | Pilot batch, LeRobot v3.0, 27.1 GB; flagged for review before training use |
| [`crossbot-coffee-serving-100-skills`](https://huggingface.co/datasets/dhruv1672/crossbot-coffee-serving-100-skills) | 100 complete runs from 200 randomised attempts, cut into 400 skill episodes (92,670 frames) | Three cameras (bird's-eye + both wrists), 15 fps |

### Isaac Sim model of the Intel RealSense D435i &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/isaac-sim-depth-camera-d435i)

A documentation-informed depth-camera surrogate for Isaac Sim 6 and ROS 2 Jazzy, developed natively on a DGX Spark (ARM64). It reproduces the behaviour that matters to a policy: invalid left band and stereo shadows, noise applied in disparity space (so error grows with distance squared), the 15 mm RGB-to-depth offset and an organised point cloud, all published on the same topics as the real driver.

<p align="center">
  <img src="assets/d435i-outputs.jpg" width="92%" alt="RGB, observation depth, aligned depth, ground-truth depth, error map and point cloud from the simulated D435i"/>
  <br/><sub>One timestamp-matched frame from every output stream, with the per-pixel error against ground truth.</sub>
</p>

### SmolVLA for SO-101 peg-in-hole insertion (Isaac Lab) &nbsp;·&nbsp; [model](https://huggingface.co/dhruv1672/smolvla-so101-peg-M6-recovery-vision-trained)

Fine-tuned `lerobot/smolvla_base` on scripted-expert demonstrations of an SO-101 arm inserting a 20 mm peg into a blind hole at 0.5 to 2.5 mm clearance, then studied what actually moves closed-loop success: chunk-relative actions, a trainable vision encoder, recovery demonstrations and serving-time filters.

<p align="center">
  <img src="assets/so101-peg-insertion.jpg" width="92%" alt="Overhead and wrist camera frames of the SO-101 arm approaching and inserting a peg"/>
  <br/><sub>Overhead and wrist cameras from an expert demonstration in the training data.</sub>
</p>

| Policy (development scenes, 80 episodes each) | Success |
|---|---|
| M6, raw actions | 23 / 80 |
| M6 + constant-velocity Kalman filter on arm targets | **30 / 80** |
| Scripted expert, 0.5 to 1.0 mm clearance | 30 / 30 |

Not yet a reliable insertion controller; the model cards keep the negative results (such as fine-tuning on extra tight-clearance data making things worse) next to the positive ones.

### Simwright: an AI copilot for robot simulation &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/simwright)

An MCP server and terminal copilot that builds and runs scenes in Isaac Sim, Isaac Lab, ROS 2, Gazebo and MuJoCo, then checks the result (physics stability, renders, reward curves, the live ROS graph) and repairs what broke. Works from Claude Code, Codex, Cursor, VS Code or ChatGPT.

```text
you   > Build a kitchen with a Unitree G1 standing at the table, utensils on it, and check the physics
agent > imports the G1 URDF, tunes drives, places SimReady assets on the tabletop, runs 3 s of physics,
        captures a frame, reports "83 bodies stable, nothing off the table"
```

### Spartan StudyBuddy: on-premise onboarding tutor &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/spartan-studybuddy) · [model](https://huggingface.co/dhruv1672/spartan-studybuddy-qwen3.8-27b-nvfp4)

Turns a team's private repositories and documents into a role-specific onboarding curriculum with a Socratic tutor, running entirely on the team's own hardware. The tutor is Qwen3.8-27B with a LoRA on the last 16 layers, merged, quantised to NVFP4 (51.75 GiB → 18.73 GiB) and served with vLLM using the model's native multi-token-prediction head as a speculative drafter.

| DGX Spark (GB10), batch 1 | Decode tok/s (p50) | Draft acceptance |
|---|---|---|
| NVFP4, no speculation | 12.72 | n/a |
| NVFP4 + native MTP speculation | **18.61 (+46%)** | 74.1% |

### GRIDPILOT: autonomous on-call crew for a GPU cloud &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/gridpilot)

Hackathon build: agents that detect and diagnose data-centre incidents, simulate remediation options, phone the on-call engineer for approval when policy requires a human, roll back their own failed fixes from git, and hold suspicious vendor invoices with cited reasons.

<p align="center">
  <img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/gif-thermal-incident.gif" width="92%" alt="A rack overheats and GRIDPILOT opens the incident on its own"/>
  <br/><sub>A rack overheats and the incident opens with no human in the loop.</sub>
</p>

### Domain-style LoRA for Stable Diffusion 1.5 &nbsp;·&nbsp; DATA 266 coursework · [model](https://huggingface.co/dhruv1672/data266-pokemon-lora)

A config-driven sweep over LoRA rank (4 to 128) and alpha with cosine decay, EMA and min-SNR weighting; versions are tagged on Hugging Face and never overwritten. The same course project included a multimodal hybrid RAG system and a five-agent LangGraph image pipeline running on open-weight models.

<p align="center">
  <img src="assets/pokemon-lora-base-vs-r64.jpg" width="80%" alt="Four prompts generated by the base model (top) and the rank-64 LoRA (bottom)"/>
  <br/><sub>Same prompt and seed, base model (top) vs final LoRA (bottom). First four in-domain test prompts where neither image was blocked by the safety checker; not hand-picked.</sub>
</p>

| Held-out test (20 prompts × 2 seeds) | KID ↓ | Blinded human preference for style |
|---|---|---|
| SD 1.5 base | 0.0648 | n/a |
| + LoRA r64, α 64 | **0.0203** | **59 / 75 (78.7%)** over base, 3 raters |

---

## Earlier this year: real robots in the lab

<table>
<tr>
<td width="50%" valign="top">
<img src="assets/go2-semantic-nav-runtime.jpg" alt="ROS 2 terminal and RViz occupancy map while the Go2 navigates"/>
<br/><b>Unitree Go2 semantic navigation</b> · <a href="https://github.com/pateldhruv1672/go2_ros">go2_ros</a><br/>
<sub>ROS 2 Jazzy workspace: SLAM, Nav2, semantic teach-and-resume navigation and an agentic skill layer on the wireless Go2.</sub>
</td>
<td width="50%" valign="top">
<img src="assets/xarm-teleop-rgbd-dashboard.jpg" alt="xArm 7 control dashboard with live D435i RGB, depth and segmentation streams"/>
<br/><b>xArm 7 teleoperation and data capture</b> · <a href="https://github.com/pateldhruv1672/xarm-teleop">xarm-teleop</a> · <a href="https://github.com/pateldhruv1672/X-ARM-Teleop-meta-Quest">Quest teleop</a><br/>
<sub>Browser and VR teleoperation with live RGB, depth, segmentation and IMU over WebRTC, for demonstration datasets.</sub>
</td>
</tr>
</table>

Video walkthroughs of both are on the [portfolio](https://pateldhruv1672.github.io/).

## Data and software projects

| Project | What it does | Stack |
|---|---|---|
| [Avian Journeys](https://github.com/pateldhruv1672/Data-Visualization) | Analysis of 10,000+ migratory journeys: geospatial flight paths, weather effects, Power BI story | Python, Pandas, Power BI |
| [THE LAST DEGREE](https://github.com/pateldhruv1672/THE-LAST-DEGREE) | Degree ROI and skill demand from job-listing data | Adzuna API, Snowflake, dbt |
| [AutoQuote](https://github.com/pateldhruv1672/auto-quote) | Photo of vehicle damage → shop discovery → AI phone calls for quotes → booking | Vision API, VAPI, Leaflet |
| [Pinecone Search Engine](https://github.com/pateldhruv1672/Pinecone-Search-Engine) | End-to-end semantic search pipeline | SentenceTransformers, Pinecone, Airflow, Docker |
| [Stock forecasting](https://github.com/pateldhruv1672/Stock-Price-Forecasting-using-Snowflake-and-Airflow) | ETL, transformations and 14-day forecasts | Airflow, Snowflake, dbt |

<details>
<summary><b>Screenshots</b></summary>
<br/>
<p align="center">
  <img width="88%" alt="Avian Journeys dashboard" src="https://github.com/user-attachments/assets/503e28c0-115a-481e-898c-b9c42faa6aeb"/>
  <img width="88%" alt="THE LAST DEGREE dashboard" src="https://github.com/user-attachments/assets/d7a78023-648b-4d00-9bb7-6f53623fd86b"/>
  <img width="88%" alt="AutoQuote" src="https://github.com/user-attachments/assets/f24a3915-1a3c-4763-b03b-316555e09b21"/>
  <img width="88%" alt="Stock forecasting dashboard" src="https://github.com/user-attachments/assets/a19c960c-9351-4ab4-8cf5-5cab2c1f865e"/>
</p>
</details>

## Toolbox

<p>
  <img src="https://skillicons.dev/icons?i=python,pytorch,cpp,ros,docker,kubernetes,linux,aws,gcp,postgres,kafka,react,ts,git&theme=light" alt="Skills"/>
</p>
<p>
  <img alt="NVIDIA Isaac" src="https://img.shields.io/badge/NVIDIA_Isaac_Sim_%2F_Lab-76B900?style=flat-square&logo=nvidia&logoColor=white"/>
  <img alt="LeRobot" src="https://img.shields.io/badge/LeRobot-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-30A2FF?style=flat-square"/>
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white"/>
  <img alt="Neo4j" src="https://img.shields.io/badge/Neo4j-4581C3?style=flat-square&logo=neo4j&logoColor=white"/>
  <img alt="Snowflake" src="https://img.shields.io/badge/Snowflake-29B5E8?style=flat-square&logo=snowflake&logoColor=white"/>
  <img alt="Airflow" src="https://img.shields.io/badge/Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white"/>
  <img alt="dbt" src="https://img.shields.io/badge/dbt-FF694B?style=flat-square&logo=dbt&logoColor=white"/>
</p>

## Activity

<p align="center">
  <img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672/output/github-contribution-grid-snake.svg" alt="Contribution graph animation" />
</p>

<p align="center"><sub>Open to research collaborations and robotics / applied-AI roles. The fastest way to reach me is <a href="https://www.linkedin.com/in/dhruvpatel16/">LinkedIn</a> or <a href="mailto:dhruvkumarkamleshbhai.patel@sjsu.edu">email</a>.</sub></p>
