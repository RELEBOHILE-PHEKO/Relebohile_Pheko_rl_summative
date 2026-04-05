import pygame
import numpy as np
import sys

# Simple color palette for the dashboard
BLACK = (10, 10, 10)
DARK_GREY = (28, 28, 28)
MID_GREY = (60, 60, 60)
LIGHT_GREY = (180, 180, 180)
WHITE = (240, 240, 240)

GREEN = (0, 220, 80)
YELLOW = (255, 210, 0)
RED = (220, 50, 50)

ORANGE = (255, 140, 0)
CYAN = (0, 210, 220)
BLUE = (70, 140, 255)

# Window size
SCREEN_W, SCREEN_H = 980, 620


def vital_colour(value, low, high, margin_ratio=0.15):
    # Return a color based on how safe the value is relative to target range.
    if low <= value <= high:
        return GREEN

    margin = (high - low) * margin_ratio
    if (low - margin) <= value <= (high + margin):
        return YELLOW

    return RED


class ICURenderer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("ICU Sepsis - RL Agent Monitor")

        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()

        self.font_xl = pygame.font.SysFont("Consolas", 46, True)
        self.font_lg = pygame.font.SysFont("Consolas", 22, True)
        self.font_sm = pygame.font.SysFont("Consolas", 13)

        self.history = {k: [] for k in ["hr", "bp", "o2", "lactate", "infection"]}
        self.max_hist = 100

    def _panel(self, rect, border=MID_GREY):
        pygame.draw.rect(self.screen, DARK_GREY, rect, border_radius=8)
        pygame.draw.rect(self.screen, border, rect, 2, border_radius=8)

    def _text(self, text, font, color, x, y, anchor="topleft"):
        surface = font.render(str(text), True, color)
        self.screen.blit(surface, surface.get_rect(**{anchor: (x, y)}))

    def _vital_card(self, name, value, unit, low, high, plot_low, plot_high, x, y):
        color = vital_colour(value, low, high)

        width, height = 150, 130
        rect = pygame.Rect(x, y, width, height)
        self._panel(rect, color)

        self._text(name, self.font_sm, LIGHT_GREY, x + 8, y + 6)
        self._text(f"{value:.1f}", self.font_xl, color, x + width // 2, y + 42, "center")
        self._text(unit, self.font_sm, LIGHT_GREY, x + width // 2, y + 88, "center")
        self._text(f"{low}-{high}", self.font_sm, MID_GREY, x + 8, y + height - 16)

        bar_x, bar_y, bar_w = x + 8, y + height - 7, width - 16
        pygame.draw.rect(self.screen, MID_GREY, (bar_x, bar_y, bar_w, 5), border_radius=3)

        fill = int(np.clip((value - plot_low) / (plot_high - plot_low), 0, 1) * bar_w)
        pygame.draw.rect(self.screen, color, (bar_x, bar_y, fill, 5), border_radius=3)

    def _wave(self, history, color, x, y, width, height, label):
        self._panel(pygame.Rect(x, y, width, height))
        self._text(label, self.font_sm, color, x + 6, y + 4)

        if len(history) < 2:
            return

        arr = np.array(history[-width:], float)
        low, high = arr.min(), arr.max()

        if high - low < 0.5:
            high = low + 0.5

        norm = (arr - low) / (high - low)

        points = [
            (x + i * max(1, width // len(norm)), y + height - 6 - int(v * (height - 12)))
            for i, v in enumerate(norm)
        ]

        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 2)

    def draw(self, hr, bp, o2, lac, inf, t, action="-", status="ONGOING"):
        for key, value in zip(self.history.keys(), [hr, bp, o2, lac, inf]):
            self.history[key].append(value)
            if len(self.history[key]) > self.max_hist:
                self.history[key].pop(0)

        self.screen.fill(BLACK)

        pygame.draw.rect(self.screen, DARK_GREY, (0, 0, SCREEN_W, 42))
        pygame.draw.line(self.screen, MID_GREY, (0, 42), (SCREEN_W, 42))

        self._text("ICU SEPSIS - RL MONITOR", self.font_lg, CYAN, 14, 10)
        self._text(f"Step {int(t)}/150", self.font_lg, WHITE, SCREEN_W - 150, 10)

        self._vital_card("HR", hr, "bpm", 60, 100, 30, 180, 20, 52)
        self._vital_card("BP", bp, "mmHg", 110, 130, 50, 200, 180, 52)
        self._vital_card("O2", o2, "%", 95, 100, 70, 100, 340, 52)
        self._vital_card("LAC", lac, "mmol", 0, 2, 0, 10, 500, 52)
        self._vital_card("INF", inf, "score", 0, 2, 0, 10, 660, 52)

        box_x, box_y, box_w = 820, 62, 140
        self._panel(pygame.Rect(box_x, box_y, box_w, 120), BLUE)

        pct = int(t) / 150
        pygame.draw.rect(self.screen, MID_GREY, (box_x + 10, box_y + 30, box_w - 20, 14))
        pygame.draw.rect(self.screen, BLUE, (box_x + 10, box_y + 30, int((box_w - 20) * pct), 14))

        self._text(f"{int(pct * 100)}%", self.font_lg, BLUE, box_x + box_w // 2, box_y + 55, "center")

        self._wave(self.history["hr"], vital_colour(hr, 60, 100), 20, 196, 185, 90, "HR")
        self._wave(self.history["bp"], vital_colour(bp, 110, 130), 215, 196, 185, 90, "BP")
        self._wave(self.history["o2"], vital_colour(o2, 95, 100), 410, 196, 185, 90, "O2")
        self._wave(self.history["lactate"], vital_colour(lac, 0, 2), 605, 196, 185, 90, "Lac")
        self._wave(self.history["infection"], vital_colour(inf, 0, 2), 800, 196, 185, 90, "Inf")

        self._panel(pygame.Rect(20, 300, 580, 55), BLUE)
        self._text("ACTION:", self.font_sm, LIGHT_GREY, 32, 308)
        self._text(action, self.font_lg, BLUE, 32, 323)

        status_color = {
            "ONGOING": WHITE,
            "RECOVERED": GREEN,
            "DEATH": RED,
            "TIMEOUT": ORANGE,
        }.get(status, WHITE)

        self._panel(pygame.Rect(610, 300, 350, 55), status_color)
        self._text("STATUS:", self.font_sm, LIGHT_GREY, 622, 308)
        self._text(status, self.font_lg, status_color, 622, 323)

        pygame.display.flip()
        self.clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                self.close()
                sys.exit()

    def close(self):
        pygame.quit()
