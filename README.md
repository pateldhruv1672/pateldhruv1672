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
  <img src="assets/g1-tray-transfer.jpg" width="92%" alt="Unitree G1 (29 DoF, BrainCo Revo2 hands) transferring a tray of cups between two tables, one recorded episode (seed 2301)."/>
  <br/><sub>Unitree G1 (29 DoF, BrainCo Revo2 hands) transferring a tray of cups between two tables, one recorded episode (seed 2301).</sub>
</p>

<p align="center">
  <img src="assets/g1-head-camera.jpg" width="80%" alt="What the policy sees: the G1 head camera (simulated RealSense D435i) during the same task."/>
  <br/><sub>What the policy sees: the G1 head camera (simulated RealSense D435i) during the same task.</sub>
</p>

<p align="center">
  <img src="assets/g1-final-frames-50.jpg" width="92%" alt="Final frame of all 50 recorded episodes: the tray ends in the centre of the dining table every time (1 to 4 cups, randomised)."/>
  <br/><sub>Final frame of all 50 recorded episodes: the tray ends in the centre of the dining table every time (1 to 4 cups, randomised).</sub>
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

<p align="center">
  <img src="assets/d435i-variants.jpg" width="80%" alt="Observation-model variants on the same scene: clean, documented default (97.6% valid, 12.4 mm RMS) and a noisier disparity setting."/>
  <br/><sub>Observation-model variants on the same scene: clean, documented default (97.6% valid, 12.4 mm RMS) and a noisier disparity setting.</sub>
</p>

### SmolVLA for SO-101 peg-in-hole insertion (Isaac Lab) &nbsp;·&nbsp; [model](https://huggingface.co/dhruv1672/smolvla-so101-peg-M6-recovery-vision-trained)

Fine-tuned `lerobot/smolvla_base` on scripted-expert demonstrations of an SO-101 arm inserting a 20 mm peg into a blind hole at 0.5 to 2.5 mm clearance, then studied what actually moves closed-loop success: chunk-relative actions, a trainable vision encoder, recovery demonstrations and serving-time filters.

<p align="center">
  <img src="assets/so101-peg-insertion.jpg" width="92%" alt="Overhead and wrist cameras from an expert demonstration: approach and insertion."/>
  <br/><sub>Overhead and wrist cameras from an expert demonstration: approach and insertion.</sub>
</p>

<p align="center">
  <img src="assets/so101-peg-insertion-2.jpg" width="92%" alt="A second randomised scene (different floor, block, peg colour and lighting) from the training data."/>
  <br/><sub>A second randomised scene (different floor, block, peg colour and lighting) from the training data.</sub>
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
  <img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/gif-thermal-incident.gif" width="92%" alt="A rack overheats and the incident opens with no human in the loop."/>
  <br/><sub>A rack overheats and the incident opens with no human in the loop.</sub>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/gif-operator-option-r38.gif" width="92%" alt="The on-call engineer proposes a different rack by phone; the plan is re-simulated and measured against the forecast."/>
  <br/><sub>The on-call engineer proposes a different rack by phone; the plan is re-simulated and measured against the forecast.</sub>
</p>

<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/01-control-room.png" alt="3D control room"/><br/><sub>3D control room</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/06-agent-operations.png" alt="Agent operations and traces"/><br/><sub>Agent operations and traces</sub></td>
</tr>
</table>
<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/05-finance-held-invoice.png" alt="Invoice held with cited sources (duplicate PO, segregation of duties)"/><br/><sub>Invoice held with cited sources (duplicate PO, segregation of duties)</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/12-approval-center.png" alt="Approval centre"/><br/><sub>Approval centre</sub></td>
</tr>
</table>
<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/07-twin-reliability.png" alt="Digital-twin reliability view"/><br/><sub>Digital-twin reliability view</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/10-scenarios.png" alt="What-if scenarios"/><br/><sub>What-if scenarios</sub></td>
</tr>
</table>
<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/09-load-gpus.png" alt="GPU load"/><br/><sub>GPU load</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/gridpilot/main/docs/media/08-sustainability.png" alt="Sustainability"/><br/><sub>Sustainability</sub></td>
</tr>
</table>

### Domain-style LoRA for Stable Diffusion 1.5 &nbsp;·&nbsp; DATA 266 coursework · [model](https://huggingface.co/dhruv1672/data266-pokemon-lora)

A config-driven sweep over LoRA rank (4 to 128) and alpha with cosine decay, EMA and min-SNR weighting; versions are tagged on Hugging Face and never overwritten. The same course project included a multimodal hybrid RAG system and a five-agent LangGraph image pipeline running on open-weight models.

<p align="center">
  <img src="assets/pokemon-lora-base-vs-r64.jpg" width="80%" alt="Same prompt and seed, base model (top) vs final LoRA (bottom). First four in-domain test prompts where neither image was blocked by the safety checker; not hand-picked."/>
  <br/><sub>Same prompt and seed, base model (top) vs final LoRA (bottom). First four in-domain test prompts where neither image was blocked by the safety checker; not hand-picked.</sub>
</p>

<p align="center">
  <img src="assets/pokemon-lora-hard-prompts.jpg" width="80%" alt="Harder prompts: two compositional (left) and two out-of-domain (right). The adapter restyles creatures but leaves a tram and a landscape largely unchanged."/>
  <br/><sub>Harder prompts: two compositional (left) and two out-of-domain (right). The adapter restyles creatures but leaves a tram and a landscape largely unchanged.</sub>
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

### Unitree Go2 semantic navigation &nbsp;·&nbsp; [go2_ros](https://github.com/pateldhruv1672/go2_ros)

ROS 2 Jazzy workspace for the wireless Go2: SLAM and localisation, Nav2, semantic teach-and-resume navigation and an agentic skill layer (voice and motion skills).

<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/go2-navigation-poster.jpg" alt="Autonomous navigation pass in the lab"/><br/><sub>Autonomous navigation pass in the lab</sub></td>
<td width="50%" valign="top"><img src="assets/go2-semantic-nav-runtime.jpg" alt="Runtime: ROS 2 terminals and the live occupancy map"/><br/><sub>Runtime: ROS 2 terminals and the live occupancy map</sub></td>
</tr>
</table>
<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/go2-repeat-run-poster.jpg" alt="Repeat traversal of a taught route"/><br/><sub>Repeat traversal of a taught route</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/go2-motion-poster.jpg" alt="Motion skills"/><br/><sub>Motion skills</sub></td>
</tr>
</table>

### xArm 7 teleoperation and demonstration capture &nbsp;·&nbsp; [xarm-teleop](https://github.com/pateldhruv1672/xarm-teleop) · [Quest teleop](https://github.com/pateldhruv1672/X-ARM-Teleop-meta-Quest)

Browser and Meta Quest teleoperation of the xArm 7 with live RGB, depth, segmentation and IMU streamed over WebRTC, used to record demonstration datasets on the real arms and in Isaac Sim.

<table>
<tr>
<td width="50%" valign="top"><img src="assets/xarm-teleop-rgbd-dashboard.jpg" alt="Teleop dashboard with live D435i RGB, depth and segmentation"/><br/><sub>Teleop dashboard with live D435i RGB, depth and segmentation</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/xarm-vr-poster.jpg" alt="VR multiview perception: RGB, depth and segmentation from two viewpoints"/><br/><sub>VR multiview perception: RGB, depth and segmentation from two viewpoints</sub></td>
</tr>
</table>
<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/xarm-lab.webp" alt="The bimanual xArm 7 station"/><br/><sub>The bimanual xArm 7 station</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/xarm-isaac-poster.jpg" alt="Teleoperating the simulated arm in Isaac Sim"/><br/><sub>Teleoperating the simulated arm in Isaac Sim</sub></td>
</tr>
</table>

### Sparse 3D memory and digital-twin reconstruction

Reconstructing the lab from camera captures (COLMAP), regenerating it as a sparse voxel scene (fVDB) inside Omniverse, and using the result as spatial memory for the robots.

<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/colmap-pointcloud.png" alt="COLMAP point cloud of the lab"/><br/><sub>COLMAP point cloud of the lab</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/fvdb-regeneration-poster.jpg" alt="fVDB scene regeneration in Omniverse"/><br/><sub>fVDB scene regeneration in Omniverse</sub></td>
</tr>
</table>
<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/pointcloud-poster.jpg" alt="Point-cloud reconstruction and inspection"/><br/><sub>Point-cloud reconstruction and inspection</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/world3d-lab.webp" alt="Lab map built on the robot"/><br/><sub>Lab map built on the robot</sub></td>
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
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/avian/dashboard-119.webp" alt="Where birds are found, by habitat and species"/><br/><sub>Where birds are found, by habitat and species</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/avian/dashboard-120.webp" alt="Environmental factors and migration interruptions"/><br/><sub>Environmental factors and migration interruptions</sub></td>
</tr>
</table>
<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/avian/dashboard-121.webp" alt="Seasonal dynamics and nesting success"/><br/><sub>Seasonal dynamics and nesting success</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/avian/dashboard-118.webp" alt="Key statistics"/><br/><sub>Key statistics</sub></td>
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
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/last-degree-overview.png" alt="Graduate ROI intelligence overview"/><br/><sub>Graduate ROI intelligence overview</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/last-degree-layoff.png" alt="Layoff risk monitor"/><br/><sub>Layoff risk monitor</sub></td>
</tr>
</table>
<table>
<tr>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/last-degree-architecture.webp" alt="Architecture: Scorecard, WARN, BLS and Adzuna → Airflow → analytics mart → Superset and web app"/><br/><sub>Architecture: Scorecard, WARN, BLS and Adzuna → Airflow → analytics mart → Superset and web app</sub></td>
<td width="50%" valign="top"><img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/last-degree-dbt-dag.webp" alt="Airflow DAG running the dbt models"/><br/><sub>Airflow DAG running the dbt models</sub></td>
</tr>
</table>
<p align="center">
  <img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/last-degree-lineage.webp" width="80%" alt="dbt lineage from raw sources to the analytics mart."/>
  <br/><sub>dbt lineage from raw sources to the analytics mart.</sub>
</p>

| Goal | Approach | Outcome |
|-----------|--------------|----------|
| Analyse 1M+ job listings | Adzuna API → Snowflake | Career intelligence |
| Calculate degree ROI | dbt transformations | Data-driven decisions |
| Forecast skill demand | ML analytics | Forward-looking insights |

### AutoQuote: AI vehicle repair finder &nbsp;·&nbsp; [repo](https://github.com/pateldhruv1672/auto-quote)

<a href="https://github.com/pateldhruv1672/auto-quote">
  <p align="center">
    <img width="90%" alt="AutoQuote" src="https://github.com/user-attachments/assets/f24a3915-1a3c-4763-b03b-316555e09b21"/>
  </p>
</a>

<p align="center">
  <img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/autoquote-app.png" width="80%" alt="Upload a damage photo, see nearby repair shops on the map."/>
  <br/><sub>Upload a damage photo, see nearby repair shops on the map.</sub>
</p>

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

<p align="center">
  <img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672.github.io/main/assets/images/stock-forecasting.png" width="70%" alt="Ticker search and price history in the forecasting app."/>
  <br/><sub>Ticker search and price history in the forecasting app.</sub>
</p>

---

## Toolbox

**AI / ML**

<p>
  <img src="https://skillicons.dev/icons?i=python,pytorch,tensorflow,docker,kubernetes,aws,gcp&theme=light" alt="AI and ML"/>
</p>
<p>
  <img alt="NVIDIA Isaac" src="https://img.shields.io/badge/NVIDIA_Isaac_Sim_%2F_Lab-76B900?style=flat-square&logo=nvidia&logoColor=white"/>
  <img alt="ROS 2" src="https://img.shields.io/badge/ROS_2_Jazzy-22314E?style=flat-square&logo=ros&logoColor=white"/>
  <img alt="LeRobot" src="https://img.shields.io/badge/LeRobot-FFD21E?style=flat-square&logo=huggingface&logoColor=black"/>
  <img alt="vLLM" src="https://img.shields.io/badge/vLLM-30A2FF?style=flat-square"/>
  <img alt="LangGraph" src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white"/>
  <img alt="Neo4j" src="https://img.shields.io/badge/Neo4j-4581C3?style=flat-square&logo=neo4j&logoColor=white"/>
</p>

**Data**

<p>
  <img src="https://skillicons.dev/icons?i=postgres,mongodb,redis,kafka&theme=light" alt="Data"/>
</p>
<p>
  <img alt="Snowflake" src="https://img.shields.io/badge/Snowflake-29B5E8?style=flat-square&logo=snowflake&logoColor=white" />
  <img alt="Pinecone" src="https://img.shields.io/badge/Pinecone-1F6FEB?style=flat-square" />
  <img alt="Airflow" src="https://img.shields.io/badge/Airflow-017CEE?style=flat-square&logo=apacheairflow&logoColor=white" />
  <img alt="dbt" src="https://img.shields.io/badge/dbt-FF694B?style=flat-square&logo=dbt&logoColor=white" />
  <img alt="Power BI" src="https://img.shields.io/badge/Power_BI-F2C811?style=flat-square&logo=powerbi&logoColor=black" />
</p>

**Frontend and tooling**

<p>
  <img src="https://skillicons.dev/icons?i=react,js,ts,html,css,vite,git,linux,vscode,java,cpp&theme=light" alt="Frontend and tooling"/>
</p>

---

## Activity

<p align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=pateldhruv1672&bg_color=ffffff&color=1e3a5f&line=1e3a5f&point=0f172a&area=true&hide_border=true" alt="Contribution activity graph"/>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/pateldhruv1672/pateldhruv1672/output/github-contribution-grid-snake.svg" alt="Contribution graph animation" />
</p>

<p align="center"><sub>Open to research collaborations and robotics / applied-AI roles. The fastest way to reach me is <a href="https://www.linkedin.com/in/dhruvpatel16/">LinkedIn</a> or <a href="mailto:dhruvkumarkamleshbhai.patel@sjsu.edu">email</a>.</sub></p>
