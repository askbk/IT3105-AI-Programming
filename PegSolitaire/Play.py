from RandomAgent import RandomAgent
from Board import Board
from BoardDisplay import BoardDisplay


def main():
    agent = RandomAgent()

    boards = [Board()]
    moves = []

    while not boards[-1].is_game_finished():
        possible_actions = boards[-1].get_possible_moves()
        moves.append(agent.choose_action(boards[-1], possible_actions))
        boards.append(boards[-1].make_move(moves[-1]))

    print("Final score:", boards[-1].get_game_score())
    BoardDisplay.display_board_move_sequence(boards[1:], moves, 4)


if __name__ == "__main__":
    main()