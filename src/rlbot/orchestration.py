from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Mapping, Sequence

from .room import TABLES, TABLE_L, TABLE_W


class Target(str, Enum):
    COLORED_CUBES = 'colored_cubes'
    RED_BALL_BOX = 'red_ball_box'
    BLUE_CUBE_RECTANGLE = 'blue_cube_rectangle'


@dataclass(frozen=True)
class Recognition:
    target: Target
    confidence: float
    stamp: float


@dataclass(frozen=True)
class ToolCall:
    tool: str
    station: str


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    details: dict


ROUTES = {
    Target.COLORED_CUBES: ToolCall('run_fable', 'cubes'),
    Target.RED_BALL_BOX: ToolCall('run_act', 'ball'),
    Target.BLUE_CUBE_RECTANGLE: ToolCall('run_flybrain', 'pick'),
}


def classify_scene(objects: Sequence[tuple[str, str]], container: str) -> Target:
    objects = tuple(objects)
    if objects == (('ball', 'red'),) and container == 'box':
        return Target.RED_BALL_BOX
    if objects == (('cube', 'blue'),) and container == 'rectangle':
        return Target.BLUE_CUBE_RECTANGLE
    colors = {'red', 'green', 'blue', 'orange', 'purple', 'yellow'}
    if (container == 'box' and len(objects) >= 2
            and all(kind == 'cube' and color in colors for kind, color in objects)
            and len({color for _, color in objects}) >= 2):
        return Target.COLORED_CUBES
    raise ValueError('Unrecognized or ambiguous scene; no controller selected')


def station_at(point: Sequence[float]) -> str:
    if len(point) != 2 or not all(math.isfinite(v) for v in point):
        raise ValueError('A spot must be a finite world-frame (x, y) pair')
    matches = []
    for table in TABLES:
        dx, dy = point[0]-table.centre[0], point[1]-table.centre[1]
        c, s = math.cos(table.yaw), math.sin(table.yaw)
        along, across = c*dx+s*dy, -s*dx+c*dy
        if ((abs(along) <= TABLE_L/2 and abs(across) <= TABLE_W/2)
                or math.dist(point, table.dock[:2]) <= .15):
            matches.append(table.name.removeprefix('table_'))
    if len(matches) != 1:
        raise ValueError('Spot must identify exactly one table or docking pose')
    return matches[0]


def route_for(station: str, recognition: Recognition, *, now: float) -> ToolCall:
    if not all(math.isfinite(v) for v in (recognition.confidence, recognition.stamp, now)):
        raise ValueError('Recognition confidence and timestamps must be finite')
    if not .8 <= recognition.confidence <= 1 or not 0 <= recognition.stamp <= now <= recognition.stamp+2:
        raise ValueError('Recognition must be confident and no more than two seconds old')
    try:
        call = ROUTES[Target(recognition.target)]
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError('Unsupported recognition; ACT is not a VLA fallback') from error
    if station != call.station:
        raise ValueError(f'Recognized {recognition.target} does not match station {station!r}')
    return call


class Dispatcher:
    def __init__(self, navigate: Callable[[str], ToolResult],
                 prepare: Mapping[str, Callable[[], Callable[[], ToolResult]]]):
        self.navigate, self.prepare = navigate, dict(prepare)

    def run(self, station: str, recognition: Recognition, *, now: float) -> ToolResult:
        call = route_for(station, recognition, now=now)
        if call.tool not in self.prepare:
            raise RuntimeError(f'{call.tool} is unavailable; supply its controller and checkpoint')
        execute = self.prepare[call.tool]()
        if not callable(execute):
            raise TypeError(f'{call.tool} preflight must return an executable controller')
        navigation = self.navigate(call.station)
        if not navigation.ok:
            return ToolResult(False, dict(tool=call.tool, navigation=navigation.details,
                                          manipulation='not started'))
        manipulation = execute()
        return ToolResult(manipulation.ok, dict(tool=call.tool, navigation=navigation.details,
                                                manipulation=manipulation.details))
