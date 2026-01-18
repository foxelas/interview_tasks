import heapq
import json
import logging
import os
from collections import deque
import mistralai
from mistralai import Mistral
from sokoban import Sokoban, to_, solved, move, to_ascii

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise RuntimeError("No API key provided. Set MISTRAL_API_KEY or pass api_key.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

caption_model = "mistral-small-2506"
text_model = "mistral-large-latest"
vision_model = "pixtral-large-latest"
client = Mistral(api_key=api_key)
    
class LlmPredictor:
    def __init__(self):
        self.model = text_model

    def ask(self, message, is_json=False, system=None, history=None, **kwargs):
        history = history or []
        if system is None:
            history += [{"role": "user", "content": message}]
        else:
            history += ([{"role": "system", "content": system}] +
                        [{"role": "user", "content": message}])
        logger.debug(f'Message sent to LLM: {message}')
        logger.debug('Full conversation history:')
        for msg in history:
            logger.debug(f"{msg['role']}: {msg['content']}")

        try:
            args = {
                "model": self.model,
                "messages": history,
                "max_tokens": 1050,
                "stream": False,
                "response_format": {"type": "json_object"} if is_json else {"type": "text"},
            }
            args.update(kwargs)

            response = client.chat.complete(**args)
            content = response.choices[0].message.content
        except mistralai.models.sdkerror.SDKError as e:
            import os
            key = os.getenv("MISTRAL_API_KEY")[:8] + "***"
            content = f"LLM API is down. Key {key} \nError: {str(e)}" if not is_json else {
                "content": f"LLM API is down. Key {key} \nError: {str(e)}"}

        if is_json:
            if isinstance(content, (dict, list)):
                return content
            try:
                return json.loads(content)
            except Exception as e:
                raise ValueError(f"Expected valid JSON, got: {content}\nError: {e}")
        else:
            return content

    def score_moves(self, state_ascii, action_history=None, format_='ascii'):
        action_history = action_history or []
        history_text = ("No moves have been made yet." if not action_history
                        else f"Moves so far: {', '.join(action_history)}")

        if format_ != 'ascii':
            state_board = json.dumps(to_(state_ascii, format_))
        else:
            state_board = state_ascii

        prompt = (
            "Given this Sokoban board and action history, score each possible move ('up', 'down', 'left', 'right') for its promise to lead toward solving the puzzle. "
            "Assign a score between 0 (very bad move) and 1 (very promising move) to each. "
            "Respond ONLY with a JSON object like: {\"up\": 0.2, \"down\": 0.5, \"left\": 0.1, \"right\": 0.9}.\n"
            "Current board:\n"
            f"{state_board}\n"
            f"\nAction history: {history_text}\n"
        )
        response = self.ask(prompt, is_json=True, temperature=0.3)
        if isinstance(response, dict):
            return {k: float(max(0, min(1, float(v)))) for k, v in response.items() if
                    k in {'up', 'down', 'left', 'right'}}
        return {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}

    def predict(self, state_ascii, action_history=None, format_='ascii'):
        """
        action_history: a list of previous moves (e.g., ['up','up','left'])
        """
        action_history = action_history or []
        history_text = ("No moves have been made yet." if not action_history
                        else f"Moves so far: {', '.join(action_history)}")

        if format_ != 'ascii':
            state_board = json.dumps(to_(state_ascii, format_))
        else:
            state_board = state_ascii

        prompt = (
            f"Given this Sokoban board, which single move (up, down, left, or right) should the player take next?\n"
            f"Legend: # wall, @ player, $ box, . goal, * box on goal, + player on goal, ' ' floor\n"
            f"The goal is for the player ('@') to push all boxes ('$') onto the goal positions ('.') with the fewest moves possible, while avoiding getting stuck at a wall or corner.\n"
            "Current board:\n"
            f"{state_board}\n"
            f"\nAction history: {history_text}\n"
            f"Your response should be ONLY a JSON object with the key 'move' and the value being one of the four directions 'up', 'down', 'left', or 'right'."
        )
        response = self.ask(prompt, is_json=True, temperature=0.5)
        move_ = response.get("move").lower() if isinstance(response, dict) else None
        return move_

    def heuristic(self, state_ascii, action_history=None, format_='ascii'):
        action_history = action_history or []
        history_text = ("No moves have been made yet." if not action_history
                        else f"Moves so far: {', '.join(action_history)}")

        if format_ != 'ascii':
            state_board = json.dumps(to_(state_ascii, format_))
        else:
            state_board = state_ascii

        prompt = (
            "Given the Sokoban board below and action history, estimate how close the board is to being solved. "
            "Return a value between 0 (solved) and 1 (very far/not solved). "
            "Respond ONLY with a JSON object: {\"score\": float}.\n"
            "Current board:\n"
            f"{state_board}\n"
            f"Action history: {history_text}\n"
        )
        response = self.ask(prompt, is_json=True, temperature=0.3)
        if isinstance(response, dict):
            score = response.get("score", None)
            try:
                score = float(score)
                return min(max(score, 0.0), 1.0)
            except Exception:
                return 0.5
        return 0.5  # Default in case of bad parse

def greedy_search_with_llm(env, llm_predictor, max_steps=20, format_='ascii'):
    visited = set()
    state = Sokoban(to_ascii(env.board))
    moves = []
    for step in range(max_steps):
        board_str = to_ascii(state.board)
        if solved(state.board):
            logger.info(f"Solved in {len(moves)} moves! Moves: {moves}")
            return moves
        if board_str in visited:
            logger.info(f"Cycle detected at step {step}.")
            return None
        visited.add(board_str)
        next_move = llm_predictor.predict(board_str, moves, format_=format_)
        logger.info(f"Step {step}: LLM suggests '{next_move}'")
        result = move(state.board, next_move)
        if not result:
            logger.info(f"Move '{next_move}' not possible.")
            return None
        state = Sokoban(to_ascii(result))
        moves.append(next_move)
    logger.info("Step limit exceeded.")
    return None

def beam_search_with_llm(env, llm_predictor, max_depth=10, beam_width=3, format_='ascii'):
    visited = set()
    State = lambda b, m: (Sokoban('\n'.join(''.join(row) for row in b)), list(m))
    logger.info("Starting LLM-guided search...")

    failed_count = 0
    beam = deque([State(env.board, env.moves)])
    for depth in range(max_depth):
        logger.info(f"Depth {depth}, Beam size: {len(beam)}")
        new_beam = []
        for state, moves in beam:
            board_str = to_ascii(state.board)
            if solved(state.board):
                logger.info(f"Solved in {len(moves)} moves: {moves}")
                return moves
            if board_str in visited:
                continue
            visited.add(board_str)

            action = llm_predictor.predict(board_str, moves, format_=format_)
            logger.info(f'LLM suggested action: {action}')
            if action not in ('up', 'down', 'left', 'right'):
                failed_count += 1
                if failed_count >= 5:
                    logger.warning(f"LLM failed to provide valid moves {failed_count} times. Aborting search.")
                    return None
                continue
            new_env = Sokoban(board_str)
            result = move(new_env.board, action)
            if result:
                new_env.board = result
                new_env.moves = moves + [action]
                new_beam.append(State(new_env.board, new_env.moves))
                logger.debug(f"New state after action: {to_(new_env.board, format_)}")
        beam = deque(sorted(new_beam, key=lambda s: len(s[1]))[:beam_width])
    return None

def beam_search_with_llm_scoring(env, llm_predictor, max_depth=10, beam_width=3, top_k_moves=2, format_='ascii'):
    visited = set()
    llm_score_cache = {}
    State = lambda b, m: (Sokoban('\n'.join(''.join(row) for row in b)), list(m))
    logger.info("Starting LLM-guided search...")

    beam = deque([State(env.board, env.moves)])
    for depth in range(max_depth):
        candidates = []
        for state, moves in beam:
            board_str = to_ascii(state.board)
            state_key = (board_str, tuple(moves))
            if solved(state.board):
                logger.info(f"Solved in {len(moves)} moves: {moves}")
                return moves
            if state_key in visited:
                continue
            visited.add(state_key)
            if state_key in llm_score_cache:
                move_scores = llm_score_cache[state_key]
            else:
                move_scores = llm_predictor.score_moves(board_str, moves, format_=format_)
                llm_score_cache[state_key] = move_scores
            legal_moves_scores = []
            for move_dir in ['up', 'down', 'left', 'right']:
                result = move(state.board, move_dir)
                if result:
                    legal_moves_scores.append((move_dir, move_scores.get(move_dir, 0)))
            for move_dir, score in sorted(legal_moves_scores, key=lambda x: -x[1])[:top_k_moves]:
                logger.info(f"Step {depth}: LLM suggests '{move_dir} with score {score}'")
                result = move(state.board, move_dir)
                new_env = Sokoban(to_ascii(result))
                new_env.moves = moves + [move_dir]
                candidates.append((score, new_env.board, new_env.moves))
        candidates.sort(reverse=True, key=lambda x: x[0])
        beam = deque([State(board, moves) for (_, board, moves) in candidates[:beam_width]])
        if not beam:
            break
    return None

def a_star_with_llm(env, llm_predictor, max_steps=20, format_='ascii'):
    g_score = {}
    visited = set()
    heap = []     # Heap: (f(n), h(n), steps, board_string, moves)
    state = env
    moves = []
    board_str = to_ascii(state.board)
    h = llm_predictor.heuristic(board_str, moves)
    f = 0 + h
    heapq.heappush(heap, (f, h, 0, board_str, moves))
    g_score[board_str] = 0

    while heap and max_steps > 0:
        f, h, steps, board_str, moves = heapq.heappop(heap)
        max_steps -= 1
        current_state = Sokoban(board_str)
        current_state.moves = list(moves)
        if solved(current_state.board):
            logger.info(f"Solved in {len(moves)} moves! Moves: {moves}")
            return moves
        visited.add(board_str)
        for move_dir in ['up', 'down', 'left', 'right']:
            result = move(current_state.board, move_dir)
            if result:
                new_board_str = to_ascii(result)
                if new_board_str in visited:
                    continue
                new_g = steps + 1
                if new_board_str in g_score and new_g >= g_score[new_board_str]:
                    continue
                new_h = llm_predictor.heuristic(new_board_str, moves + [move_dir], format_=format_)
                logger.info(f"Step {steps}: LLM suggests '{move_dir} with score {new_h}'")
                new_f = new_g + new_h
                g_score[new_board_str] = new_g
                heapq.heappush(heap, (new_f, new_h, new_g, new_board_str, moves + [move_dir]))
    logger.info("Failed to solve within step/visit limit.")
    return None

def load_microban(filename, N=1):
    boards = []
    with open(filename) as f:
        buf = []
        for line in f:
            line = line.rstrip('\n')
            if line.startswith(';'):
                if buf:
                    boards.append('\n'.join(buf))
                    buf = []
                continue
            if line.strip() == '':
                continue
            buf.append(line)
        if buf:
            boards.append('\n'.join(buf))
    return boards[:N]


def evaluate(func, llm_predict, N=1, microban_path='microban.txt', format_='ascii'):
    puzzles = load_microban(microban_path, N)
    results = []
    for idx, ascii_board in enumerate(puzzles):
        print(f"Solving puzzle {idx+1}/{N}...")
        print(ascii_board)
        print("-------------------")
        env = Sokoban(ascii_board)
        solution = func(env, llm_predict, format_=format_)
        solved = solution is not None
        print(f"Puzzle {idx+1}: {'Solved' if solved else 'Failed'} Steps: {len(solution) if solution else 'N/A'}")
        results.append({"solved": solved, "steps": len(solution) if solved else None})
    print(f"{sum(1 for r in results if r['solved'])}/{N} solved")
    return results

if __name__ == "__main__":
    predictor = LlmPredictor()
    N = 1
    for board_mode in [ 'structured', 'ascii']:
        print(f'Evaluating Sokoban solving with board mode: {board_mode}')
        print("\n\n[Baseline] Evaluating Greedy Search with LLM:")
        evaluate(greedy_search_with_llm, predictor, N, format_=board_mode)
        print("\n\n[Beam Search] Evaluating Beam Search with LLM:")
        evaluate(beam_search_with_llm, predictor, N, format_=board_mode)
        print("\n\n[Beam Search] Evaluating Beam Search with LLM Scoring:")
        evaluate(beam_search_with_llm_scoring, predictor, N, format_=board_mode)
        print("\n\n[A* Search] Evaluating A* Search with LLM:")
        evaluate(a_star_with_llm, predictor, N, format_=board_mode)