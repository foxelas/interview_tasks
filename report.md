# Sokoban Solving with LLM-Guided Tree Search

## 1. Code Implementation

### Sokoban Environment Parser and Simulator
- The code uses a `Sokoban` class and supporting functions (`to_`, `solved`, `move`, `to_ascii`) to parse and simulate Sokoban puzzle boards.
- Board representations are loaded from text files (e.g., `microban.txt`), parsed, and handled in both ASCII and structured formats for flexibility.

### LLM Integration for Action Prediction
- Integrates Mistral LLM via the `LlmPredictor` class for zero-shot prediction of the next move, move scoring, and heuristic assessments.
- LLM prompts are designed to guide the model towards actionable outputs (move directions and scores), supporting both textual and JSON output.
- The API key is securely retrieved from the environment, and failures are gracefully reported.

### Tree Search Algorithm Implementation
- Multiple search strategies are implemented:
  - **Greedy Search**: Uses the LLM to choose moves sequentially.
  - **Beam Search**: Expands several best candidates at each search depth using LLM predictions.
  - **Beam Search with LLM Scoring**: Scores possible moves with the LLM and expands the top-k.
  - **A* Search with LLM Heuristic**: Uses the LLM to estimate progress-to-solution for effective pruning.
- All search loops carefully avoid cycles and heuristics are cached as needed.

### Evaluation Framework
- The `evaluate` function runs experiments on N Sokoban puzzles, reporting solution quality and step count.
- Supports batch puzzle loading, flexible board representations, and systematic strategy comparison.

## 2. Experimental Analysis

### Comparing State Representations for LLM
- Two main representations are tested: 'ascii' and 'structured'.
- ASCII provides a human-readable, compact format, while structured serialization can better support LLM tokenization and reasoning.
- Empirical results in the printed output highlight performance differences between representations.

### Effect of Different Search Strategies
- The provided framework runs and compares Greedy, Beam, Beam+Scoring, and A* Search methods.
- Beam/A* searches generally outperform Greedy on harder puzzles by balancing exploration and exploitation around LLM predictions.
- Logging and output enables direct comparison of success rates and step counts.

### LLM Prediction Quality and Solving Performance
- The success of each strategy depends strongly on the quality and stability of LLM predictions and scoring.
- When LLM predictions are low-confidence or incorrect, search stagnates or cycles; more sophisticated search (beam, scoring, heuristic) can partially compensate.
- Caching/correcting failed predictions and using stricter prompt formatting improves reliability.

### Success Rate vs Puzzle Complexity
- As puzzle complexity increases, success rates for greedy approaches drop.

## 3. Discussion

### Handling Single-Step LLM Usage (Constraint)
- Each search step queries the LLM for a *single* move or score, following a *one-step look-ahead* paradigm.
- This design keeps LLM usage minimal and cost-effective but can lead to local minima (getting stuck) on complex puzzles if the LLM's local moves don't offer global foresight.

### Computational Trade-offs of Search Strategy
- **Greedy Search:** Fast, low LLM/API usage, but easily stuck on suboptimal choices due to lack of exploration.
- **Beam & A* Searches:** Increased memory and compute costs; more LLM calls per step, but much better performance for difficult puzzles.
- **LLM Scoring and Heuristics:** Moderate API usage can lead to much-improved decision boundaries; caching can reduce redundant calls.

### Future System Improvements
- **More LLM Calls:** Allowing multi-step lookahead (e.g., planning multiple moves) can greatly improve search power, albeit at a higher cost.
- **Different Architectures:** prompt engineering (e.g., chain-of-thought) could yield better move proposals.
- **Memory/Experience Replay:** Using search histories and experience for improved prompt context and error correction.
