from dataclasses import dataclass


@dataclass
class Bbox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def intersects(self, bbox) -> bool:
        return not (
            self.xmin > bbox.xmax
            or self.xmax < bbox.xmin
            or self.ymin > bbox.ymax
            or self.ymax < bbox.ymin
        )

    def intersection(self, bbox):
        if not self.intersects(bbox):
            return None
        xmin = max(self.xmin, bbox.xmin)
        ymin = max(self.ymin, bbox.ymin)
        xmax = min(self.xmax, bbox.xmax)
        ymax = min(self.ymax, bbox.ymax)
        return Bbox(xmin, ymin, xmax, ymax)

    @property
    def area(self) -> int:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def normalize_bbox(self, width, height):
        return Bbox(
            int(1000 * (self.xmin / width)),
            int(1000 * (self.ymin / height)),
            int(1000 * (self.xmax / width)),
            int(1000 * (self.ymax / height)),
        )

    def from_list(self, bbox):
        return Bbox(bbox[0], bbox[1], bbox[2], bbox[3])

    def to_list(self) -> list:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class Word:
    bbox: Bbox
    text: str
    label: str = None
