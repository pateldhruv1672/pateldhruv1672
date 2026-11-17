<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg"/>
    <source media="(prefers-color-scheme: light)" srcset="assets/banner-light.svg"/>
    <img src="assets/banner-light.svg" width="100%" alt="Dhruv Patel, Robotics and Applied AI Research Engineer"/>
  </picture>
</p>

<p align="center">
  <a href="https://pateldhruv1672.github.io/"><img src="https://img.shields.io/badge/Portfolio-pateldhruv1672.github.io-0f172a?style=flat-square" alt="Portfolio"/></a>
  <a href="https://www.linkedin.com/in/dhruvpatel16/"><img src="https://img.shields.io/badge/LinkedIn-dhruvpatel16-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn"/></a>
  <a href="https://huggingface.co/dhruv1672"><img src="https://img.shields.io/badge/Hugging%20Face-dhruv1672-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"/></a>
  <a href="mailto:dhruvkumarkamleshbhai.patel@sjsu.edu"><img src="https://img.shields.io/badge/Email-SJSU-EA4335?style=flat-square&logo=gmail&logoColor=white" alt="Email"/></a>
  <img src="https://komarev.com/ghpvc/?username=pateldhruv1672&style=flat-square&color=0f172a&label=profile+views" alt="Profile views"/>
</p>

<p align="center">
  <a href="https://git.io/typing-svg"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=16&duration=3200&pause=900&color=79C0FF&center=true&vCenter=true&width=720&lines=Robot+learning+in+simulation+%E2%86%92+hardware;Multi-robot+agents+with+shared+memory;Evidence+first%3A+every+number+is+measured"/>
    <img src="https://readme-typing-svg.herokuapp.com?font=JetBrains+Mono&size=16&duration=3200&pause=900&color=1E3A5F&center=true&vCenter=true&width=720&lines=Robot+learning+in+simulation+%E2%86%92+hardware;Multi-robot+agents+with+shared+memory;Evidence+first%3A+every+number+is+measured" alt="Robot learning in simulation to hardware; multi-robot agents with shared memory; evidence first"/>
  </picture></a>
</p>

I am a graduate researcher in the **Robotics & Digital Twin Lab at San José State University** (M.S. Applied Data Science). My work sits where robot learning, simulation and agentic AI meet: building the environments, datasets and policies that let robots do useful multi-step tasks, and measuring honestly how well they do them. Before graduate school I shipped production retrieval and ML systems at **Bain & Company** and distributed data services at **Samsung Research**.

```python
class DhruvPatel(Researcher, Engineer):
    """Robot learning, from simulation to hardware."""

    role = "Graduate Researcher, Robotics & Digital Twin Lab @ SJSU"
    degree = "M.S. Applied Data Science"
    experience = [
        "Bain & Company · Software Engineer II",
        "Samsung R&D Institute India · Software Engineer",
    ]
    focus = [
        "robot learning", "VLA fine-tuning",
        "multi-robot agents", "sim-to-real evaluation",
    ]
    robots = ["Unitree Go2", "xArm 7", "Unitree G1 (sim)", "SO-101 (sim)"]
    stack = ["Isaac Sim", "Isaac Lab", "ROS 2", "PyTorch", "LeRobot", "vLLM"]

    def principle(self) -> str:
        return "Report what was measured, including the negative results."
```

---

## Recent work (Sep to Nov 2026)

### Cross-Bot: a multi-robot service system &nbsp;·&nbsp; M.S. thesis, in progress

A bimanual xArm 7 coffee station, a Unitree G1 that carries trays between tables and a Go2 that explores and navigates. Each robot runs its own LangGraph agent; they coordinate through a shared Neo4j memory and durable handover records. Everything runs in Isaac Sim today, designed so the same stack moves to the real robots. The code stays private while the thesis is in progress; the datasets are public.

<p align="center">
  <img src="assets/g1-tray-transfer.jpg" width="92%" alt="Unitree G1 (29 DoF, BrainCo Revo2 hands) transferring a tray of cups between two tables, one recorded episode (seed 2301)."/>
  <br/><sub>Unitree G1 (29 DoF, BrainCo Revo2 hands) transferring a tray of cups between two tables, one recorded episode (seed 2301).</sub>
</p>

<p align="center">
  <img src="assets/g1-head-camera.jpg" width="80%" alt="What the policy sees: the G1 head camera (simulated RealSense D435i) during the same task."/>
  <br/><sub>What the policy sees: the G1 head camera (simulated RealSense D435i) during the same task.</sub>
</p>

<p align="center">
  <img src="assets/xarm-coffee-serving.jpg" width="92%" alt="Bimanual xArm 7 coffee serving, bird's-eye camera: cup to the outlet, brew, handover to the tray arm."/>
  <br/><sub>Bimanual xArm 7 coffee serving, bird's-eye camera: cup to the outlet, brew, handover to the tray arm.</sub>
</p>

<p align="center">
  <img src="assets/xarm-coffee-wrist.jpg" width="92%" alt="Left wrist camera through the same run: empty cup, under the outlet, brewed, set down for the handover."/>
  <br/><sub>Left wrist camera through the same run: empty cup, under the outlet, brewed, set down for the handover.</sub>
</p>

| Output | Contents | Notes |
|---|---|---|
| [`crossbot-g1-tray-delivery-rgbd`](https://huggingface.co/datasets/dhruv1672/crossbot-g1-tray-delivery-rgbd) | 50 episodes, 64,741 frames at 10 Hz, head D435i RGB + lossless depth, 51-D state / 44-D action | Pilot batch, LeRobot v3.0, 27.1 GB; flagged for review before training use |
| [`crossbot-coffee-serving-100-skills`](https://huggingface.co/datasets/dhruv1672/crossbot-coffee-serving-100-skills) | 100 complete runs from 200 randomised attempts, cut into 400 skill episodes (92,670 frames) | Three cameras (bird's-eye + both wrists), 15 fps |

---

### Isaac Sim model of the Intel RealSense D435i &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/isaac-sim-depth-camera-d435i)

A documentation-informed depth-camera surrogate for Isaac Sim 6 and ROS 2 Jazzy, developed natively on a DGX Spark (ARM64). It reproduces the behaviour that matters to a policy: invalid left band and stereo shadows, noise applied in disparity space (so error grows with distance squared), the 15 mm RGB-to-depth offset and an organised point cloud, all published on the same topics as the real driver.

<p align="center">
  <img src="assets/d435i-desktop.jpg" width="92%" alt="Running on the DGX Spark: RViz with the organised point cloud and depth/RGB previews (left), Isaac Sim through the simulated colour camera (right)."/>
  <br/><sub>Running on the DGX Spark: RViz with the organised point cloud and depth/RGB previews (left), Isaac Sim through the simulated colour camera (right).</sub>
</p>

<p align="center">
  <img src="assets/d435i-outputs.jpg" width="92%" alt="One timestamp-matched frame from every output stream, with the per-pixel error against ground truth."/>
  <br/><sub>One timestamp-matched frame from every output stream, with the per-pixel error against ground truth.</sub>
</p>

---

### SmolVLA for SO-101 peg-in-hole insertion (Isaac Lab) &nbsp;·&nbsp; [model](https://huggingface.co/dhruv1672/smolvla-so101-peg-M6-recovery-vision-trained)

Fine-tuned `lerobot/smolvla_base` on scripted-expert demonstrations of an SO-101 arm inserting a 20 mm peg into a blind hole at 0.5 to 2.5 mm clearance, then studied what actually moves closed-loop success: chunk-relative actions, a trainable vision encoder, recovery demonstrations and serving-time filters.

<p align="center">
  <img src="assets/so101-peg-insertion-2.jpg" width="92%" alt="Overhead and wrist cameras from an expert demonstration: the peg is lined up over the hole and inserted. Scenes are randomised (floor, block, peg colour, lighting)."/>
  <br/><sub>Overhead and wrist cameras from an expert demonstration: the peg is lined up over the hole and inserted. Scenes are randomised (floor, block, peg colour, lighting).</sub>
</p>

| Policy (development scenes, 80 episodes each) | Success |
|---|---|
| M6, raw actions | 23 / 80 |
| M6 + constant-velocity Kalman filter on arm targets | **30 / 80** |
| Scripted expert, 0.5 to 1.0 mm clearance | 30 / 30 |

Not yet a reliable insertion controller; the model cards keep the negative results (such as fine-tuning on extra tight-clearance data making things worse) next to the positive ones.

---

### Simwright: an AI copilot for robot simulation &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/simwright)

An MCP server and terminal copilot that builds and runs scenes in Isaac Sim, Isaac Lab, ROS 2, Gazebo and MuJoCo, then checks the result (physics stability, renders, reward curves, the live ROS graph) and repairs what broke. Works from Claude Code, Codex, Cursor, VS Code or ChatGPT.

```text
you   > Build a kitchen with a Unitree G1 standing at the table, utensils on it, and check the physics
agent > imports the G1 URDF, tunes drives, places SimReady assets on the tabletop, runs 3 s of physics,
        captures a frame, reports "83 bodies stable, nothing off the table"
```

---

### Spartan StudyBuddy: on-premise onboarding tutor &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/spartan-studybuddy) · [model](https://huggingface.co/dhruv1672/spartan-studybuddy-qwen3.8-27b-nvfp4)

Turns a team's private repositories and documents into a role-specific onboarding curriculum with a Socratic tutor, running entirely on the team's own hardware. The tutor is Qwen3.8-27B with a LoRA on the last 16 layers, merged, quantised to NVFP4 (51.75 GiB → 18.73 GiB) and served with vLLM using the model's native multi-token-prediction head as a speculative drafter.

| DGX Spark (GB10), batch 1 | Decode tok/s (p50) | Draft acceptance |
|---|---|---|
| NVFP4, no speculation | 12.72 | n/a |
| NVFP4 + native MTP speculation | **18.61 (+46%)** | 74.1% |

---

### GRIDPILOT: autonomous on-call crew for a GPU cloud &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/gridpilot)

Hackathon build: agents that detect and diagnose data-centre incidents, simulate remediation options, phone the on-call engineer for approval when policy requires a human, roll back their own failed fixes from git, and hold suspicious vendor invoices with cited reasons.

<p align="center">
  <img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/gif-thermal-incident.gif" width="92%" alt="A rack overheats and the incident opens with no human in the loop."/>
  <br/><sub>A rack overheats and the incident opens with no human in the loop.</sub>
</p>

<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/05-finance-held-invoice.png" alt="Invoice held with cited sources (duplicate PO, segregation of duties)"/><br/><sub>Invoice held with cited sources (duplicate PO, segregation of duties)</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/06-agent-operations.png" alt="Agent operations and traces"/><br/><sub>Agent operations and traces</sub></td>
</tr>
</table>

---

### Domain-style LoRA for Stable Diffusion 1.5 &nbsp;·&nbsp; DATA 266 coursework · [model](https://huggingface.co/dhruv1672/data266-pokemon-lora)

A config-driven sweep over LoRA rank (4 to 128) and alpha with cosine decay, EMA and min-SNR weighting; versions are tagged on Hugging Face and never overwritten. The same course project included a multimodal hybrid RAG system and a five-agent LangGraph image pipeline running on open-weight models.

<p align="center">
  <img src="assets/pokemon-lora-base-vs-r64.jpg" width="80%" alt="Same prompt and seed, base model (top) vs final LoRA (bottom). First four in-domain test prompts where neither image was blocked by the safety checker; not hand-picked."/>
  <br/><sub>Same prompt and seed, base model (top) vs final LoRA (bottom). First four in-domain test prompts where neither image was blocked by the safety checker; not hand-picked.</sub>
</p>

<p align="center">
  <img src="assets/pokemon-lora-loss-curves.jpg" width="80%" alt="Training loss and EMA validation loss for every run of the rank/alpha sweep."/>
  <br/><sub>Training loss and EMA validation loss for every run of the rank/alpha sweep.</sub>
</p>

| Held-out test (20 prompts × 2 seeds) | KID ↓ | Blinded human preference for style |
|---|---|---|
| SD 1.5 base | 0.0648 | n/a |
| + LoRA r64, α 64 | **0.0203** | **59 / 75 (78.7%)** over base, 3 raters |

---

## Real robots in the lab

<p align="center">
  <img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/go2-cover.webp" width="88%" alt="Robotics & Digital Twin Lab, SJSU: Unitree G1, Unitree Go2 and the bimanual xArm 7 station."/>
  <br/><sub>Robotics & Digital Twin Lab, SJSU: Unitree G1, Unitree Go2 and the bimanual xArm 7 station.</sub>
</p>

---

### Unitree Go2 semantic navigation &nbsp;·&nbsp; [go2_ros](https://github.com/pateldhruv1672/go2_ros)

ROS 2 Jazzy workspace for the wireless Go2: SLAM and localisation, Nav2, semantic teach-and-resume navigation and an agentic skill layer (voice and motion skills).

<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/go2-motion-poster.jpg" alt="Go2 running motion skills in the lab"/><br/><sub>Go2 running motion skills in the lab</sub></td>
<td width="50%" valign="top"><img src="assets/go2-semantic-nav-runtime.jpg" alt="Runtime: ROS 2 terminals and the live occupancy map"/><br/><sub>Runtime: ROS 2 terminals and the live occupancy map</sub></td>
</tr>
</table>

---

### xArm 7 teleoperation and demonstration capture &nbsp;·&nbsp; [xarm-teleop](https://github.com/pateldhruv1672/xarm-teleop) · [Quest teleop](https://github.com/pateldhruv1672/X-ARM-Teleop-meta-Quest)

Browser and Meta Quest teleoperation of the xArm 7 with live RGB, depth, segmentation and IMU streamed over WebRTC, used to record demonstration datasets on the real arms and in Isaac Sim.

<table>
<tr>
<td width="50%" valign="top"><img src="assets/xarm-teleop-rgbd-dashboard.jpg" alt="Teleop dashboard with live D435i RGB, depth and segmentation"/><br/><sub>Teleop dashboard with live D435i RGB, depth and segmentation</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/xarm-vr-poster.jpg" alt="VR multiview perception: RGB, depth and segmentation from two viewpoints"/><br/><sub>VR multiview perception: RGB, depth and segmentation from two viewpoints</sub></td>
</tr>
</table>

---

### Sparse 3D memory and digital-twin reconstruction

Reconstructing the lab from camera captures (COLMAP) and loading the result into Omniverse as spatial memory for the robots.

<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/colmap-pointcloud.png" alt="COLMAP point cloud of the lab"/><br/><sub>COLMAP point cloud of the lab</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/pointcloud-poster.jpg" alt="Reconstructed point cloud loaded in Omniverse"/><br/><sub>Reconstructed point cloud loaded in Omniverse</sub></td>
</tr>
</table>

Video walkthroughs of the lab work are on the [portfolio](https://pateldhruv1672.github.io/).

---

## Featured projects

<details open>
<summary><b>Data pipelines I maintain</b></summary>
<br/>

| Pipeline | Components | Status |
|-------|-------------------|--------|
| Semantic search pipelines | SentenceTransformers → Pinecone → Airflow | Active |
| Analytics and forecasting engine | yfinance → Snowflake → dbt → dashboards | Active |
| Job-market intelligence | Adzuna → Snowflake → analytics | Active |

</details>

---

### Avian Journeys: bird migration analytics &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/Data-Visualization)

<a href="https://github.com/pateldhruv1672/Data-Visualization">
  <p align="center">
    <img width="90%" alt="Avian Journeys" src="https://github.com/user-attachments/assets/503e28c0-115a-481e-898c-b9c42faa6aeb"/>
  </p>
  <p align="center">
    <img width="90%" alt="Bird Migration Dashboard" src="https://github.com/user-attachments/assets/5a274141-fd72-4ef3-8bb0-fde7c5be41d2"/>
  </p>
</a>

<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/avian/dashboard-120.webp" alt="Environmental factors and migration interruptions"/><br/><sub>Environmental factors and migration interruptions</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/avian/dashboard-121.webp" alt="Seasonal dynamics and nesting success"/><br/><sub>Seasonal dynamics and nesting success</sub></td>
</tr>
</table>

<table>
<tr>
<td width="50%">

**Highlights**
- Analysed **10,000+ migratory journeys**
- **Geospatial flight-path** visualisation
- Weather impact on migration success
- Interactive **Power BI** storytelling dashboard
- Python EDA with Pandas, Seaborn and Matplotlib

</td>
<td width="50%">

**Stack:** `Python` `Power BI` `Pandas` `Seaborn`

| Metric | Value |
|--------|-------|
| Variables analysed | 42 features |
| Records processed | 10,000+ |
| Visualisation type | Geospatial + BI |

</td>
</tr>
</table>

---

### THE LAST DEGREE: college ROI intelligence &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/THE-LAST-DEGREE)

<a href="https://github.com/pateldhruv1672/THE-LAST-DEGREE">
  <p align="center">
    <img width="90%" alt="THE LAST DEGREE" src="https://github.com/user-attachments/assets/d7a78023-648b-4d00-9bb7-6f53623fd86b"/>
  </p>
  <p align="center">
    <img width="90%" alt="THE LAST DEGREE Dashboard" src="https://github.com/user-attachments/assets/259c306d-f0ae-4add-8b6b-2d8dabaad07a"/>
  </p>
</a>

<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/last-degree-architecture.webp" alt="Architecture: Scorecard, WARN, BLS and Adzuna → Airflow → analytics mart → Superset and web app"/><br/><sub>Architecture: Scorecard, WARN, BLS and Adzuna → Airflow → analytics mart → Superset and web app</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/last-degree-dbt-dag.webp" alt="Airflow DAG running the dbt models"/><br/><sub>Airflow DAG running the dbt models</sub></td>
</tr>
</table>

| Goal | Approach | Outcome |
|-----------|--------------|----------|
| Analyse 1M+ job listings | Adzuna API → Snowflake | Career intelligence |
| Calculate degree ROI | dbt transformations | Data-driven decisions |
| Forecast skill demand | ML analytics | Forward-looking insights |

---

### AutoQuote: AI vehicle repair finder &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/auto-quote)

<a href="https://github.com/pateldhruv1672/auto-quote">
  <p align="center">
    <img width="90%" alt="AutoQuote" src="https://github.com/user-attachments/assets/f24a3915-1a3c-4763-b03b-316555e09b21"/>
  </p>
</a>

<table>
<tr>
<td>

**Features**
- AI damage image analysis (Freepik Vision)
- Shop discovery (Yutori Research API)
- **Voice AI phone calls** to shops for quotes (VAPI)
- One-click appointment booking
- Interactive Leaflet maps

</td>
<td>

```
 damage photo ─► scan ─► find shops ─► call ─► book
     📸            🔍         🗺️          📞       📅
```

</td>
</tr>
</table>

---

### Pinecone semantic search engine &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/Pinecone-Search-Engine)

<table>
<tr>
<td width="50%">

<a href="https://github.com/pateldhruv1672/Pinecone-Search-Engine">
<img alt="Pinecone semantic search pipeline" src="https://github.com/user-attachments/assets/df1fc994-cccd-460c-953c-fd01155c4772" />
</a>

</td>
<td width="50%">

**Capabilities**
- End-to-end semantic search pipeline
- SentenceTransformers embeddings
- Pinecone vector database
- Apache Airflow orchestration

**Stack:** `Python` `Pinecone` `Airflow` `Docker`

</td>
</tr>
</table>

---

### Stock price forecasting and analytics &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/Stock-Price-Forecasting-Analysis-dbt-airflow)

<table>
<tr>
<td width="50%">

**Capabilities**
- ETL with Apache Airflow
- Snowflake data warehousing
- dbt transformations
- Snowflake ML forecasting (14-day horizon)
- Moving averages and volatility analysis

**Stack:** `Airflow` `Snowflake` `dbt` `Python`

</td>
<td width="50%">

<a href="https://github.com/pateldhruv1672/Stock-Price-Forecasting-Analysis-dbt-airflow">
 <img alt="Stock forecasting dashboard" src="https://github.com/user-attachments/assets/a19c960c-9351-4ab4-8cf5-5cab2c1f865e" />
</a>

</td>
</tr>
</table>

---

## Toolbox

| Area | Tools |
|---|---|
| **Robotics & simulation** | <img alt="NVIDIA Isaac Sim / Lab" src="https://img.shields.io/badge/NVIDIA%20Isaac%20Sim%20%2F%20Lab-76B900?style=flat-square&logo=nvidia&logoColor=white"/> <img alt="ROS 2" src="https://img.shields.io/badge/ROS%202-22314E?style=flat-square&logo=ros&logoColor=white"/> <img alt="MuJoCo" src="https://img.shields.io/badge/MuJoCo-1F2937?style=flat-square"/> <img alt="Gazebo" src="https://img.shields.io/badge/Gazebo-F58113?style=flat-square"/> <img alt="LeRobot" src="https://img.shields.io/badge/LeRobot-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/> <img alt="Omniverse" src="https://img.shields.io/badge/Omniverse-76B900?style=flat-square&logo=nvidia&logoColor=white"/> <img alt="COLMAP" src="https://img.shields.io/badge/COLMAP-374151?style=flat-square"/> |
| **ML & LLMs** | <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"/> <img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/> <img alt="vLLM" src="https://img.shields.io/badge/vLLM-30A2FF?style=flat-square"/> <img alt="Diffusers + PEFT" src="https://img.shields.io/badge/Diffusers%20%2B%20PEFT-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/> <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white"/> |
| **Agents & retrieval** | <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white"/> <img alt="LlamaIndex" src="https://img.shields.io/badge/LlamaIndex-000000?style=flat-square"/> <img alt="MCP" src="https://img.shields.io/badge/MCP-111827?style=flat-square&logo=modelcontextprotocol&logoColor=white"/> <img alt="Neo4j" src="https://img.shields.io/badge/Neo4j-4581C3?style=flat-square&logo=neo4j&logoColor=white"/> <img alt="Pinecone" src="https://img.shields.io/badge/Pinecone-1F6FEB?style=flat-square"/> <img alt="Elasticsearch" src="https://img.shields.io/badge/Elasticsearch-005571?style=flat-square&logo=elasticsearch&logoColor=white"/> |
| **Data engineering** | <img alt="Snowflake" src="https://img.shields.io/badge/Snowflake-29B5E8?style=flat-square&logo=snowflake&logoColor=white"/> <img alt="Airflow" src="https://img.shields.io/badge/Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white"/> <img alt="dbt" src="https://img.shields.io/badge/dbt-FF694B?style=flat-square"/> <img alt="Kafka" src="https://img.shields.io/badge/Kafka-231F20?style=flat-square&logo=apachekafka&logoColor=white"/> <img alt="PySpark" src="https://img.shields.io/badge/PySpark-E25A1C?style=flat-square&logo=apachespark&logoColor=white"/> <img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white"/> <img alt="MongoDB" src="https://img.shields.io/badge/MongoDB-47A248?style=flat-square&logo=mongodb&logoColor=white"/> <img alt="Redis" src="https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white"/> <img alt="Power BI" src="https://img.shields.io/badge/Power%20BI-F2C811?style=flat-square"/> |
| **Cloud & DevOps** | <img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white"/> <img alt="Kubernetes" src="https://img.shields.io/badge/Kubernetes-326CE5?style=flat-square&logo=kubernetes&logoColor=white"/> <img alt="AWS" src="https://img.shields.io/badge/AWS-232F3E?style=flat-square"/> <img alt="Google Cloud" src="https://img.shields.io/badge/Google%20Cloud-4285F4?style=flat-square&logo=googlecloud&logoColor=white"/> <img alt="Linux" src="https://img.shields.io/badge/Linux-FCC624?style=flat-square&logo=linux&logoColor=black"/> <img alt="Git" src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white"/> |
| **Languages & frontend** | <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"/> <img alt="C++" src="https://img.shields.io/badge/C%2B%2B-00599C?style=flat-square&logo=cplusplus&logoColor=white"/> <img alt="Java" src="https://img.shields.io/badge/Java-ED8B00?style=flat-square&logo=openjdk&logoColor=white"/> <img alt="TypeScript" src="https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white"/> <img alt="JavaScript" src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black"/> <img alt="React" src="https://img.shields.io/badge/React-20232A?style=flat-square&logo=react&logoColor=61DAFB"/> <img alt="Vite" src="https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white"/> <img alt="HTML5" src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white"/> <img alt="CSS" src="https://img.shields.io/badge/CSS-1572B6?style=flat-square&logo=css&logoColor=white"/> |

---

## Activity

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github-readme-activity-graph.vercel.app/graph?username=pateldhruv1672&bg_color=0d1117&color=79c0ff&line=79c0ff&point=f0f6fc&area=true&hide_border=true"/>
    <img src="https://github-readme-activity-graph.vercel.app/graph?username=pateldhruv1672&bg_color=ffffff&color=1e3a5f&line=1e3a5f&point=0f172a&area=true&hide_border=true" alt="Contribution activity graph"/>
  </picture>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672/output/github-contribution-grid-snake.svg" alt="Contribution graph animation" />
</p>

<p align="center"><sub>Open to research collaborations and robotics / applied-AI roles. The fastest way to reach me is <a href="https://www.linkedin.com/in/dhruvpatel16/">LinkedIn</a> or <a href="mailto:dhruvkumarkamleshbhai.patel@sjsu.edu">email</a>.</sub></p>
