import json
import logging
import os
from collections import deque
import mistralai
from mistralai import Mistral
from sokoban import Sokoban, to_ascii, solved, move

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
        logger.info(f'Message sent to LLM: {message}')
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

    def predict(self, state_ascii, action_history=None):
        """
        action_history: a list of previous moves (e.g., ['up','up','left'])
        """
        action_history = action_history or []
        history_text = ("No moves have been made yet." if not action_history
                        else f"Moves so far: {', '.join(action_history)}")
        prompt = (
            f"Given this Sokoban board, which single move (up, down, left, or right) should the player take next?\n"
            f"Legend: # wall, @ player, $ box, . goal, * box on goal, + player on goal, ' ' floor\n"
            f"The goal is for the player ('@') to push all boxes ('$') onto the goal positions ('.') with the fewest moves possible, while avoiding getting stuck at a wall or corner.\n"
            f"\nThe current board is represented in ASCII art as follows:\n"
            f"{state_ascii}\n"
            f"\nAction history: {history_text}\n"
            f"Your response should be ONLY a JSON object with the key 'move' and the value being one of the four directions 'up', 'down', 'left', or 'right'."
        )
        response = self.ask(prompt, is_json=True, temperature=0.5)
        move_ = response.get("move").lower() if isinstance(response, dict) else None
        return move_

def search_with_llm(env, llm_predict, max_depth=50, beam_width=3):
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

            action = llm_predict(board_str, moves)
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
                logger.debug(f"New state after action: {to_ascii(new_env.board)}")
        beam = deque(sorted(new_beam, key=lambda s: len(s[1]))[:beam_width])
    return None  # failed

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

def evaluate(llm_predict, N=1, microban_path='microban.txt'):
    puzzles = load_microban(microban_path, N)
    results = []
    for idx, ascii_board in enumerate(puzzles):
        print(f"Solving puzzle {idx+1}/{N}...")
        print(ascii_board)
        print("-------------------")
        env = Sokoban(ascii_board)
        solution = search_with_llm(env, llm_predict)
        solved = solution is not None
        print(f"Puzzle {idx+1}: {'Solved' if solved else 'Failed'} Steps: {len(solution) if solution else 'N/A'}")
        results.append({"solved": solved, "steps": len(solution) if solved else None})
    print(f"{sum(1 for r in results if r['solved'])}/{N} solved")
    return results

if __name__ == "__main__":
    predictor = LlmPredictor()
    evaluate(predictor.predict, N=5)