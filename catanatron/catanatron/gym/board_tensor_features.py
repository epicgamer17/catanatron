import networkx as nx
import numpy as np
from typing import Literal, Optional, Dict, Tuple

from catanatron.state_functions import (
    get_player_buildings,
    get_longest_road_length,
    player_key,
)
from catanatron.models.player import Color
from catanatron.game import Game
from catanatron.models.enums import (
    RESOURCES,
    SETTLEMENT,
    CITY,
    ROAD,
    ActionType,
)
from catanatron.models.decks import RESOURCE_FREQDECK_INDEXES
from catanatron.models.coordinate_system import offset_to_cube
from catanatron.models.board import STATIC_GRAPH
from catanatron.models.map import number_probability
from catanatron.features import get_feature_ordering, iter_players

Coordinate = Tuple[int, int, int]

WIDTH = 21
HEIGHT = 11
AXIAL_WIDTH = 11
AXIAL_HEIGHT = 7

MAX_BANK_RESOURCES = 19
MAX_ROAD_LENGTH = 15

NODE_ID_MAP: Optional[Dict[int, Tuple[int, int]]] = None
EDGE_MAP: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None
TILE_COORDINATE_MAP: Optional[Dict[Coordinate, Tuple[int, int]]] = None
AXIAL_NODE_MAP: Optional[Dict[int, Tuple[int, int]]] = None
AXIAL_EDGE_MAP: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None
VALIDITY_MASK: Optional[Dict[int, np.ndarray]] = None
AXIAL_VALIDITY_MASK: Optional[Dict[int, np.ndarray]] = None


def is_graph_feature(feature_name):
    if (
        feature_name.startswith("TILE")
        or feature_name.startswith("PORT")
        or feature_name.startswith("NODE")
        or feature_name.startswith("EDGE")
    ):
        return True

    if (
        feature_name == "DICE"
        or feature_name == "IS_DISCARDING"
        or feature_name == "IS_MOVING_ROBBER"
        or feature_name.startswith("BANK_")
        or feature_name.endswith("_HAS_ROLLED")
        or feature_name.endswith("_LONGEST_ROAD_LENGTH")
    ):
        return True

    return False


def get_numeric_features(num_players):
    features = get_feature_ordering(num_players)
    return [f for f in features if not is_graph_feature(f)]


NUMERIC_FEATURES = get_numeric_features(4)
NUM_NUMERIC_FEATURES = len(NUMERIC_FEATURES)


def get_channels(
    num_players: int,
    include_validity_mask: bool = False,
    include_last_roll: bool = False,
    include_game_phase: bool = False,
    include_bank_state: bool = False,
    include_road_distance: bool = False,
) -> int:
    """
    Returns total channels:
    - 3*n (settlement, city, road per player)
    - 6 resources
    - 10 dice numbers
    - 1 robber
    - 6 ports
    - 1 validity mask (optional)
    - 11 last roll (optional)
    - 2 + n game phase (IS_DISCARDING, IS_MOVING_ROBBER, P{i}_HAS_ROLLED) (optional)
    - 5 bank counts + 5 bank empty indicators (optional)
    - n distance to longest road (optional)
    """
    base = num_players * 3 + 6 + 10 + 1 + 6
    if include_validity_mask:
        base += 1
    if include_last_roll:
        base += 11
    if include_game_phase:
        base += 2 + num_players
    if include_bank_state:
        base += 10
    if include_road_distance:
        base += num_players
    return base


def get_node_and_edge_maps() -> (
    Tuple[Dict[int, Tuple[int, int]], Dict[Tuple[int, int], Tuple[int, int]]]
):
    global NODE_ID_MAP, EDGE_MAP
    if NODE_ID_MAP is None or EDGE_MAP is None:
        NODE_ID_MAP, EDGE_MAP = init_board_tensor_map()
    return NODE_ID_MAP, EDGE_MAP


def get_tile_coordinate_map() -> Dict[Coordinate, Tuple[int, int]]:
    global TILE_COORDINATE_MAP
    if TILE_COORDINATE_MAP is None:
        TILE_COORDINATE_MAP = init_tile_coordinate_map()
    return TILE_COORDINATE_MAP


def get_validity_mask(game_map) -> np.ndarray:
    global VALIDITY_MASK
    if VALIDITY_MASK is None:
        VALIDITY_MASK = {}

    map_size = len(game_map.land_nodes)
    if map_size not in VALIDITY_MASK:
        VALIDITY_MASK[map_size] = init_validity_mask(game_map)
    return VALIDITY_MASK[map_size]


def get_axial_node_edge_maps() -> (
    Tuple[Dict[int, Tuple[int, int]], Dict[Tuple[int, int], Tuple[int, int]]]
):
    global AXIAL_NODE_MAP, AXIAL_EDGE_MAP
    if AXIAL_NODE_MAP is None or AXIAL_EDGE_MAP is None:
        AXIAL_NODE_MAP, AXIAL_EDGE_MAP = init_axial_maps()
    return AXIAL_NODE_MAP, AXIAL_EDGE_MAP


def get_axial_validity_mask(game_map) -> np.ndarray:
    global AXIAL_VALIDITY_MASK
    if AXIAL_VALIDITY_MASK is None:
        AXIAL_VALIDITY_MASK = {}

    map_size = len(game_map.land_nodes)
    if map_size not in AXIAL_VALIDITY_MASK:
        AXIAL_VALIDITY_MASK[map_size] = init_axial_validity_mask(game_map)
    return AXIAL_VALIDITY_MASK[map_size]


def init_board_tensor_map():
    pairs = [
        (82, 93),
        (79, 94),
        (42, 25),
        (41, 26),
        (73, 59),
        (72, 60),
    ]
    paths = [nx.shortest_path(STATIC_GRAPH, a, b) for (a, b) in pairs]

    node_map = {}
    edge_map = {}
    for i, path in enumerate(paths):
        for j, node in enumerate(path):
            node_map[node] = (2 * j, 2 * i)

            node_has_down_edge = (i + j) % 2 == 0
            if node_has_down_edge and i + 1 < len(pairs):
                next_path = paths[i + 1]
                edge_map[(node, next_path[j])] = (2 * j, 2 * i + 1)
                edge_map[(next_path[j], node)] = (2 * j, 2 * i + 1)

            if j + 1 < len(path):
                edge_map[(node, path[j + 1])] = (2 * j + 1, 2 * i)
                edge_map[(path[j + 1], node)] = (2 * j + 1, 2 * i)

    return node_map, edge_map


def init_tile_coordinate_map():
    tile_map = {}

    width_step = 4
    height_step = 2
    for i in range(HEIGHT // height_step):
        for j in range(WIDTH // width_step):
            (offset_x, offset_y) = (-2 + j, -2 + i)
            cube_coordinate = offset_to_cube((offset_x, offset_y))

            maybe_odd_offset = (i % 2) * 2
            tile_map[cube_coordinate] = (
                height_step * i,
                width_step * j + maybe_odd_offset,
            )
    return tile_map


def init_validity_mask(game_map):
    node_map, edge_map = get_node_and_edge_maps()
    tile_map = get_tile_coordinate_map()

    mask = np.zeros((WIDTH, HEIGHT), dtype=np.float32)

    for node_id in game_map.land_nodes:
        if node_id in node_map:
            x, y = node_map[node_id]
            mask[x, y] = 1.0

    for tile in game_map.land_tiles.values():
        for edge_id in tile.edges.values():
            if edge_id in edge_map:
                x, y = edge_map[edge_id]
                mask[x, y] = 1.0

        if tile.id in tile_map:
            y, x = tile_map[tile.id]
            for dx in [0, 2, 4]:
                for dy in [0, 2]:
                    mask[x + dx, y + dy] = 1.0

    return mask


def init_axial_maps():
    node_map = {}
    edge_map = {}

    pairs = [
        (82, 93),
        (79, 94),
        (42, 25),
        (41, 26),
        (73, 59),
        (72, 60),
    ]
    paths = [nx.shortest_path(STATIC_GRAPH, a, b) for (a, b) in pairs]

    for path_idx, path in enumerate(paths):
        for step_idx, node in enumerate(path):
            node_map[node] = (step_idx * 2, path_idx * 2)

            if step_idx + 1 < len(path):
                next_node = path[step_idx + 1]
                edge_mid_x = step_idx * 2 + 1
                edge_map[(node, next_node)] = (edge_mid_x, path_idx * 2)
                edge_map[(next_node, node)] = (edge_mid_x, path_idx * 2)

            if (path_idx + step_idx) % 2 == 0 and path_idx + 1 < len(paths):
                next_path = paths[path_idx + 1]
                edge_map[(node, next_path[step_idx])] = (step_idx * 2, path_idx * 2 + 1)
                edge_map[(next_path[step_idx], node)] = (step_idx * 2, path_idx * 2 + 1)

    return node_map, edge_map


def init_axial_validity_mask(game_map):
    node_map, edge_map = get_axial_node_edge_maps()

    width = AXIAL_WIDTH * 2
    height = AXIAL_HEIGHT * 2
    mask = np.zeros((width, height), dtype=np.float32)

    for node_id in game_map.land_nodes:
        if node_id in node_map:
            x, y = node_map[node_id]
            ix, iy = int(x), int(y)
            if 0 <= ix < width and 0 <= iy < height:
                mask[ix, iy] = 1.0

    for tile in game_map.land_tiles.values():
        for edge_id in tile.edges.values():
            if edge_id in edge_map:
                x, y = edge_map[edge_id]
                ix, iy = int(x), int(y)
                if 0 <= ix < width and 0 <= iy < height:
                    mask[ix, iy] = 1.0

        xs = [node_map[n][0] for n in tile.nodes.values() if n in node_map]
        ys = [node_map[n][1] for n in tile.nodes.values() if n in node_map]
        if xs and ys:
            x_avg = sum(xs) / len(xs)
            y_avg = sum(ys) / len(ys)
            ix, iy = int(round(x_avg)), int(round(y_avg))
            if 0 <= ix < width and 0 <= iy < height:
                mask[ix, iy] = 1.0

    return mask


SpatialEncoding = Literal["sparse", "axial"]


def create_board_tensor(
    game: Game,
    p0_color: Color,
    channels_first: bool = False,
    spatial_encoding: SpatialEncoding = "sparse",
    include_validity_mask: bool = False,
    include_last_roll: bool = False,
    include_game_phase: bool = False,
    include_bank_state: bool = False,
    include_road_distance: bool = False,
):
    """Creates a board tensor with configurable spatial encoding.

    Channel layout:
        - 3*n player planes (settlement, city, road per player)
        - 6 resource planes (binary: Wood, Ore, Sheep, Wheat, Brick, Desert)
        - 10 dice number planes (binary: 2-6, 8-12)
        - 1 robber plane
        - 6 port planes
        - Optional channels (in order):
          - 1 validity mask
          - 11 last roll (one-hot)
          - 2 + n game phase (IS_DISCARDING, IS_MOVING_ROBBER, P{i}_HAS_ROLLED)
          - 10 bank state (5 normalized counts + 5 empty indicators)
          - n distance to longest road

    Args:
        game: The Catan game state
        p0_color: The color of the perspective player
        channels_first: If True, shape is (C, H, W), else (H, W, C)
        spatial_encoding: "sparse" for 21x11 grid, "axial" for dense hex grid
        include_validity_mask: If True, appends a binary validity mask channel
        include_last_roll: If True, appends 11 one-hot channels for last dice roll
        include_game_phase: If True, appends game phase binary planes
        include_bank_state: If True, appends bank resource count and empty planes
        include_road_distance: If True, appends distance to longest road planes
    """
    n = len(game.state.colors)

    if spatial_encoding == "axial":
        width = AXIAL_WIDTH * 2
        height = AXIAL_HEIGHT * 2
        node_map, edge_map = get_axial_node_edge_maps()
    else:
        width = WIDTH
        height = HEIGHT
        node_map, edge_map = get_node_and_edge_maps()

    channels = get_channels(
        n,
        include_validity_mask,
        include_last_roll,
        include_game_phase,
        include_bank_state,
        include_road_distance,
    )
    planes = np.zeros((channels, width, height), dtype=np.float32)

    for i, color in iter_players(tuple(game.state.colors), p0_color):
        settlement_channel = 3 * i
        city_channel = 3 * i + 1
        road_channel = 3 * i + 2

        for node_id in get_player_buildings(game.state, color, SETTLEMENT):
            if node_id in node_map:
                x, y = node_map[node_id]
                planes[settlement_channel, int(x), int(y)] = 1.0

        for node_id in get_player_buildings(game.state, color, CITY):
            if node_id in node_map:
                x, y = node_map[node_id]
                planes[city_channel, int(x), int(y)] = 1.0

        for edge in get_player_buildings(game.state, color, ROAD):
            if edge in edge_map:
                x, y = edge_map[edge]
                planes[road_channel, int(x), int(y)] = 1.0

    base_channel = 3 * n
    resources = list(RESOURCES)

    if spatial_encoding == "sparse":
        tile_map = get_tile_coordinate_map()

        for coordinate, tile in game.state.board.map.land_tiles.items():
            (y, x) = tile_map[coordinate]

            if tile.resource is not None:
                resource_channel_idx = base_channel + resources.index(tile.resource)
            else:
                resource_channel_idx = base_channel + 5

            for dx in [0, 2, 4]:
                for dy in [0, 2]:
                    planes[resource_channel_idx, x + dx, y + dy] = 1.0

            if tile.number is not None:
                if tile.number <= 6:
                    dice_idx = tile.number - 2
                else:
                    dice_idx = tile.number - 3
                dice_channel_idx = base_channel + 6 + dice_idx

                for dx in [0, 2, 4]:
                    for dy in [0, 2]:
                        planes[dice_channel_idx, x + dx, y + dy] = 1.0

        (y, x) = tile_map[game.state.board.robber_coordinate]
        robber_channel_idx = base_channel + 6 + 10
        for dx in [0, 2, 4]:
            for dy in [0, 2]:
                planes[robber_channel_idx, x + dx, y + dy] = 1.0

        for resource, node_ids in game.state.board.map.port_nodes.items():
            channel_idx_delta = 5 if resource is None else resources.index(resource)
            channel_idx = base_channel + 6 + 10 + 1 + channel_idx_delta
            for node_id in node_ids:
                if node_id in node_map:
                    (x, y) = node_map[node_id]
                    planes[channel_idx, x, y] = 1.0

    elif spatial_encoding == "axial":
        for coordinate, tile in game.state.board.map.land_tiles.items():
            xs = [node_map[n][0] for n in tile.nodes.values() if n in node_map]
            ys = [node_map[n][1] for n in tile.nodes.values() if n in node_map]

            if not xs or not ys:
                continue

            x_avg = sum(xs) / len(xs)
            y_avg = sum(ys) / len(ys)

            x = int(round(x_avg))
            y = int(round(y_avg))

            if tile.resource is not None:
                resource_channel_idx = base_channel + resources.index(tile.resource)
            else:
                resource_channel_idx = base_channel + 5

            planes[resource_channel_idx, x, y] = 1.0

            if tile.number is not None:
                if tile.number <= 6:
                    dice_idx = tile.number - 2
                else:
                    dice_idx = tile.number - 3
                dice_channel_idx = base_channel + 6 + dice_idx
                planes[dice_channel_idx, x, y] = 1.0

        robber_tile = game.state.board.map.land_tiles.get(
            game.state.board.robber_coordinate
        )
        if robber_tile is not None:
            xs = [node_map[n][0] for n in robber_tile.nodes.values() if n in node_map]
            ys = [node_map[n][1] for n in robber_tile.nodes.values() if n in node_map]
            if xs and ys:
                x_avg = sum(xs) / len(xs)
                y_avg = sum(ys) / len(ys)
                x_robber = int(round(x_avg))
                y_robber = int(round(y_avg))
                robber_channel_idx = base_channel + 6 + 10
                planes[robber_channel_idx, x_robber, y_robber] = 1.0

        for resource, node_ids in game.state.board.map.port_nodes.items():
            channel_idx_delta = 5 if resource is None else resources.index(resource)
            channel_idx = base_channel + 6 + 10 + 1 + channel_idx_delta
            for node_id in node_ids:
                if node_id in node_map:
                    (x, y) = node_map[node_id]
                    planes[channel_idx, int(x), int(y)] = 1.0

    current_channel = base_channel + 6 + 10 + 1 + 6

    if include_validity_mask:
        if spatial_encoding == "axial":
            mask = get_axial_validity_mask(game.state.board.map)
        else:
            mask = get_validity_mask(game.state.board.map)
        planes[current_channel] = mask[:width, :height]
        current_channel += 1

    if include_last_roll:
        last_roll = getattr(game.state, "last_roll", None)
        if last_roll is not None and len(last_roll) == 2:
            total = sum(last_roll)
            if 2 <= total <= 12:
                roll_idx = total - 2
                planes[current_channel + roll_idx, :, :] = 1.0
        current_channel += 11

    if include_game_phase:
        is_discarding = any(
            action.action_type == ActionType.DISCARD for action in game.playable_actions
        )
        is_moving_robber = any(
            action.action_type == ActionType.MOVE_ROBBER
            for action in game.playable_actions
        )

        if is_discarding:
            planes[current_channel, :, :] = 1.0
        current_channel += 1

        if is_moving_robber:
            planes[current_channel, :, :] = 1.0
        current_channel += 1

        for i, color in iter_players(tuple(game.state.colors), p0_color):
            key = player_key(game.state, color)
            has_rolled = game.state.player_state.get(key + "_HAS_ROLLED", False)
            if has_rolled:
                planes[current_channel, :, :] = 1.0
            current_channel += 1

    if include_bank_state:
        bank_freqdeck = game.state.resource_freqdeck
        if bank_freqdeck is None:
            bank_freqdeck = [0, 0, 0, 0, 0]

        for res_idx, resource in enumerate(resources):
            freq_idx = RESOURCE_FREQDECK_INDEXES.get(resource, res_idx)
            count = bank_freqdeck[freq_idx] if freq_idx < len(bank_freqdeck) else 0
            normalized = count / MAX_BANK_RESOURCES
            planes[current_channel, :, :] = normalized
            current_channel += 1

        for res_idx, resource in enumerate(resources):
            freq_idx = RESOURCE_FREQDECK_INDEXES.get(resource, res_idx)
            count = bank_freqdeck[freq_idx] if freq_idx < len(bank_freqdeck) else 0
            if count == 0:
                planes[current_channel, :, :] = 1.0
            current_channel += 1

    if include_road_distance:
        road_lengths = {}
        for color in game.state.colors:
            road_lengths[color] = get_longest_road_length(game.state, color)

        max_road = max(road_lengths.values()) if road_lengths else 0

        for i, color in iter_players(tuple(game.state.colors), p0_color):
            my_road = road_lengths.get(color, 0)
            if max_road > 0:
                distance = (max_road - my_road) / MAX_ROAD_LENGTH
            else:
                distance = 0.0
            planes[current_channel, :, :] = distance
            current_channel += 1

    if not channels_first:
        return np.transpose(planes, (1, 2, 0))
    return planes


def get_spatial_dims(spatial_encoding: SpatialEncoding) -> Tuple[int, int]:
    """Returns (width, height) for the given spatial encoding."""
    if spatial_encoding == "axial":
        return (AXIAL_WIDTH * 2, AXIAL_HEIGHT * 2)
    return (WIDTH, HEIGHT)
