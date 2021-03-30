from functools import reduce
from networkx.algorithms.operators import union
from networkx import Graph
from networkx.drawing.nx_pylab import draw, draw_networkx_nodes, draw_networkx_edges
from networkx.drawing.layout import rescale_layout
from Hex.Game import Board
import matplotlib.pyplot as plt
import numpy as np


class BoardDisplay:
    """
    Class for displaying boards.
    """

    @staticmethod
    def _rotate(arr: np.array) -> np.array:
        """
        Rotates the matrix arr by -135 degrees.
        Input must be an nx2 matrix.
        Returns an nx2 matrix
        """
        return np.dot(np.array([[-0.707, 0.707], [-0.707, -0.707]]), arr.T).T

    @staticmethod
    def _calculate_node_positions(board_positions):
        """
        Calculates how nodes should be positioned when drawing the board.
        """
        position_matrix = np.array(
            [list(position) for position in board_positions]
        ).astype("float64")

        rotated = BoardDisplay._rotate(position_matrix)

        return dict(zip(board_positions, rotated))

    @staticmethod
    def _draw_nodes(graph, node_positions, nodes, node_size, node_color):
        """
        Draw the nodes of the board.
        """
        draw_networkx_nodes(
            graph,
            pos=node_positions,
            nodelist=nodes,
            node_size=node_size,
            node_color=node_color,
        )

    @staticmethod
    def _draw_edges(graph, node_positions):
        """
        Draws the edges of the board.
        """
        draw_networkx_edges(graph, pos=node_positions)

    @staticmethod
    def _draw_holes(graph, node_positions, holes):
        """
        Draws the holes on the board.
        """
        BoardDisplay._draw_nodes(graph, node_positions, holes, 75, "black")

    @staticmethod
    def _draw_pieces(graph, node_positions, player_1_pieces, player_2_pieces):
        """
        Draws the game pieces on the board.
        """
        BoardDisplay._draw_nodes(graph, node_positions, player_1_pieces, 100, "blue")
        BoardDisplay._draw_nodes(graph, node_positions, player_2_pieces, 100, "red")

    @staticmethod
    def _display_board(board):
        """
        Displays a board state while emphasising the previous move.
        """
        edges = board.get_edge_list()
        graph = Graph(edges)
        holes = board.get_all_holes()
        player_1_pieces, player_2_pieces = board.get_all_pieces()
        node_positions = BoardDisplay._calculate_node_positions(
            holes + player_1_pieces + player_2_pieces
        )

        BoardDisplay._draw_edges(graph, node_positions)
        BoardDisplay._draw_holes(graph, node_positions, holes)
        BoardDisplay._draw_pieces(
            graph, node_positions, player_1_pieces, player_2_pieces
        )

        plt.draw()
        plt.axis("off")

    @staticmethod
    def display_board_sequence(boards, pause=1):
        """
        Displays a sequence of boards
        """
        for board in boards:
            plt.figure(1)
            plt.clf()
            BoardDisplay._display_board(board)
            plt.pause(pause)
            plt.clf()


if __name__ == "__main__":
    board = Board(size=4)
    moves = [(2, 0), (2, 2), (2, 1), (3, 1), (1, 2), (1, 3), (0, 3)]
    boards = reduce(
        lambda history, move: [*history, history[-1].make_move(move)], moves, [board]
    )
    BoardDisplay.display_board_sequence(boards)
