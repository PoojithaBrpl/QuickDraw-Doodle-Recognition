import os
import numpy as np
import pygame
import tensorflow as tf

# Model settings
MODEL_PATH = "models/quickdraw_cnn.keras"
CLASSES = [
    "apple", "banana", "car", "cat", "dog",
    "fish", "house", "star", "tree", "airplane"
]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("Loading CNN model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# Smaller window that fits the screen better
pygame.init()
WIDTH, HEIGHT = 850, 620
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("QuickDraw Doodle Recognition")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (55, 115, 210)
LIGHT_GRAY = (225, 225, 225)
DARK = (35, 35, 45)

font = pygame.font.SysFont("Arial", 25)
small_font = pygame.font.SysFont("Arial", 19)
title_font = pygame.font.SysFont("Arial", 27, bold=True)

# Drawing area
CANVAS_X, CANVAS_Y = 20, 70
CANVAS_SIZE = 480
BRUSH_SIZE = 10

canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
canvas.fill(BLACK)

predict_button = pygame.Rect(530, 230, 280, 55)
clear_button = pygame.Rect(530, 300, 280, 55)

prediction = "Draw something!"
confidence = ""
drawing = False

def draw_button(rect, text):
    pygame.draw.rect(screen, BLUE, rect, border_radius=10)
    label = font.render(text, True, WHITE)
    screen.blit(label, label.get_rect(center=rect.center))

def predict_doodle():
    # Convert the canvas to grayscale.
    pixels = pygame.surfarray.array3d(canvas)
    pixels = np.transpose(pixels, (1, 0, 2))
    grayscale = pixels.mean(axis=2)

    # Find the area containing the drawing.
    rows, cols = np.where(grayscale > 20)

    if len(rows) == 0:
        return ["Draw something first!"]

    # Crop around the drawing.
    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1
    cropped = grayscale[top:bottom, left:right]

    # Add padding so the drawing is not cut off.
    h, w = cropped.shape
    side = max(h, w) + 20
    square = np.zeros((side, side), dtype="float32")

    y = (side - h) // 2
    x = (side - w) // 2
    square[y:y+h, x:x+w] = cropped

    # Resize to the CNN input size.
    small = pygame.surfarray.make_surface(
        np.repeat(square[:, :, None], 3, axis=2).astype("uint8")
    )
    small = pygame.transform.smoothscale(small, (28, 28))

    pixels = pygame.surfarray.array3d(small)
    pixels = np.transpose(pixels, (1, 0, 2))
    grayscale = pixels.mean(axis=2)

    image_array = (grayscale / 255.0).astype("float32")
    image_array = image_array.reshape(1, 28, 28, 1)

    probabilities = model.predict(image_array, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]

    result_lines = []
    for index in top_indices:
        result_lines.append(
            f"{CLASSES[index]}: {probabilities[index] * 100:.1f}%"
        )

    return result_lines

running = True

while running:
    screen.fill(WHITE)

    heading = title_font.render("QuickDraw Doodle Recognition", True, DARK)
    screen.blit(heading, (20, 20))

    screen.blit(canvas, (CANVAS_X, CANVAS_Y))
    pygame.draw.rect(
        screen, LIGHT_GRAY,
        (CANVAS_X, CANVAS_Y, CANVAS_SIZE, CANVAS_SIZE), 3
    )

    side_title = font.render("Prediction", True, DARK)
    screen.blit(side_title, (530, 90))

    result = small_font.render(prediction, True, DARK)
    screen.blit(result, (530, 135))

    confidence_label = small_font.render(confidence, True, DARK)
    screen.blit(confidence_label, (530, 170))

    draw_button(predict_button, "Predict")
    draw_button(clear_button, "Clear")

    instructions = [
        "How to use:",
        "1. Draw inside the black box.",
        "2. Click Predict.",
        "3. Click Clear to try again."
    ]

    for i, line in enumerate(instructions):
        text = small_font.render(line, True, DARK)
        screen.blit(text, (530, 400 + i * 30))

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos

            if predict_button.collidepoint(x, y):
                results = predict_doodle()
                prediction = results[0]
                confidence = " | ".join(results[1:])

            elif clear_button.collidepoint(x, y):
                canvas.fill(BLACK)
                prediction = "Draw something!"
                confidence = ""

            elif (
                CANVAS_X <= x < CANVAS_X + CANVAS_SIZE
                and CANVAS_Y <= y < CANVAS_Y + CANVAS_SIZE
            ):
                drawing = True
                pygame.draw.circle(
                    canvas, WHITE,
                    (x - CANVAS_X, y - CANVAS_Y), BRUSH_SIZE
                )

        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False

        elif event.type == pygame.MOUSEMOTION and drawing:
            x, y = event.pos
            local_x = x - CANVAS_X
            local_y = y - CANVAS_Y

            if 0 <= local_x < CANVAS_SIZE and 0 <= local_y < CANVAS_SIZE:
                pygame.draw.circle(
                    canvas, WHITE, (local_x, local_y), BRUSH_SIZE
                )
                previous_x = local_x - event.rel[0]
                previous_y = local_y - event.rel[1]

                pygame.draw.line(
                    canvas, WHITE,
                    (previous_x, previous_y),
                    (local_x, local_y),
                    BRUSH_SIZE * 2
                )

pygame.quit()
