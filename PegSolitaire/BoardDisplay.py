from networkx.algorithms.operators import union
from networkx import Graph
from networkx.drawing.nx_pylab import draw, draw_networkx_nodes, draw_networkx_edges
from networkx.drawing.layout import rescale_layout
from Board import Board
from Position import Position
from typing import Tuple, List, Dict
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np


class BoardDisplay:
    """
    Class for displaying board transitions.
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
    def _calculate_node_positions(
        board_positions: List[Position],
    ) -> Dict[Position, np.array]:
        """
        Calculates how nodes should be positioned when drawing the board.
        """
        position_matrix = np.array(
            list(
                map(lambda position: list(position.get_coordinates()), board_positions)
            )
        ).astype("float64")

        rotated = BoardDisplay._rotate(position_matrix)

        return dict(zip(board_positions, rotated))

    @staticmethod
    def _draw_holes(graph, node_positions, holes):
        """
        Draws the holes on the board.
        """
        draw_networkx_nodes(
            graph, pos=node_positions, nodelist=holes, node_size=75, node_color="black"
        )

    @staticmethod
    def _draw_edges(graph, node_positions):
        """
        Draws the edges of the board.
        """
        draw_networkx_edges(graph, pos=node_positions)

    @staticmethod
    def _draw_pieces(graph, node_positions, pieces):
        """
        Draws the game pieces on the board.
        """
        draw_networkx_nodes(
            graph, pos=node_positions, nodelist=pieces, node_color="blue"
        )

    @staticmethod
    def _draw_move(graph, node_positions, move: Tuple[Position]):
        """
        Draws the board transition caused by a move.
        """
        from_position, to_position = move

        draw_networkx_nodes(
            graph,
            pos=node_positions,
            nodelist=[from_position],
            node_color="red",
            node_size=75,
        )
        draw_networkx_nodes(
            graph,
            pos=node_positions,
            nodelist=[to_position],
            node_color="red",
            node_size=500,
        )

    @staticmethod
    def display_board_with_move(board: Board, move: Tuple[Position]):
        """
        Displays a board state while emphasising the previous move.
        """
        edges = board.get_edge_list()
        graph = Graph(edges)
        holes = board.get_all_holes()
        pieces = board.get_all_pieces()
        node_positions = BoardDisplay._calculate_node_positions(holes + pieces)

        BoardDisplay._draw_edges(graph, node_positions)
        BoardDisplay._draw_holes(graph, node_positions, holes)
        BoardDisplay._draw_pieces(graph, node_positions, pieces)
        BoardDisplay._draw_move(graph, node_positions, move)

        plt.draw()
        plt.axis("off")
        plt.savefig(fname="board.png")


BoardDisplay.display_board_with_move(
    Board(
        size=5, shape="triangle", hole_positions=[Position((1, 2)), Position((1, 3))]
    ),
    (Position((1, 3)), Position((1, 1))),
)
