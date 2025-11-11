#include <gtk/gtk.h>
#include <gdk/gdk.h>
#include "gann.h"
#include <stdio.h>
#include "utils.h"
#include <math.h>

// --- Constants ---
#define CANVAS_WIDTH 280
#define CANVAS_HEIGHT 280
#define GRID_SIZE 28
#define CELL_SIZE (CANVAS_WIDTH / GRID_SIZE)
#define NETWORK_INPUT_SIZE (GRID_SIZE * GRID_SIZE)
const char* NETWORK_FILE = "trained_network.dat";

// --- Global Variables ---
static int grid[GRID_SIZE][GRID_SIZE] = {0};
static GtkWidget *drawing_area;
static GtkWidget *prediction_label;
static GtkWidget *model_status_label;
static NeuralNetwork* net = NULL;

// --- Function Prototypes ---
static void clear_grid();
static void process_and_predict();
static void load_network(const char* filename);
static void load_model_button_clicked(GtkWidget *widget, gpointer data);
static void save_grid_as_pgm(const char* filename, double* data);
static void preprocess_and_center_image(double* network_input);

// --- GUI Callbacks ---

static void load_network(const char* filename) {
    if (net) {
        nn_free(net);
        net = NULL;
    }

    net = nn_load(filename);

    if (net) {
        char status_text[1024];
        g_snprintf(status_text, sizeof(status_text), "Model: %s", g_path_get_basename(filename));
        gtk_label_set_text(GTK_LABEL(model_status_label), status_text);
    } else {
        gtk_label_set_text(GTK_LABEL(model_status_label), "Error: Failed to load model.");
    }
}

static void load_model_button_clicked(GtkWidget *widget, gpointer data) {
    GtkWidget *dialog;
    GtkFileChooserAction action = GTK_FILE_CHOOSER_ACTION_OPEN;
    gint res;

    dialog = gtk_file_chooser_dialog_new("Open File",
                                         GTK_WINDOW(data),
                                         action,
                                         "_Cancel",
                                         GTK_RESPONSE_CANCEL,
                                         "_Open",
                                         GTK_RESPONSE_ACCEPT,
                                         NULL);

    GtkFileFilter *filter = gtk_file_filter_new();
    gtk_file_filter_set_name(filter, "Network files (*.dat)");
    gtk_file_filter_add_pattern(filter, "*.dat");
    gtk_file_chooser_add_filter(GTK_FILE_CHOOSER(dialog), filter);

    res = gtk_dialog_run(GTK_DIALOG(dialog));
    if (res == GTK_RESPONSE_ACCEPT) {
        char *filename;
        GtkFileChooser *chooser = GTK_FILE_CHOOSER(dialog);
        filename = gtk_file_chooser_get_filename(chooser);
        load_network(filename);
        g_free(filename);
    }

    gtk_widget_destroy(dialog);
}

/**
 * @brief Clears the grid to all white.
 */
static void clear_grid() {
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            grid[i][j] = 0;
        }
    }
    gtk_widget_queue_draw(drawing_area);
}

/**
 * @brief Callback for the "Clear" button.
 */
static void clear_button_clicked(GtkWidget *widget, gpointer data) {
    clear_grid();
    gtk_label_set_text(GTK_LABEL(prediction_label), "Prediction: -");
}

/**
 * @brief Callback for the "Predict" button.
 */
static void predict_button_clicked(GtkWidget *widget, gpointer data) {
    process_and_predict();
}

/**
 * @brief Redraw the grid on the canvas.
 */
static gboolean draw_cb(GtkWidget *widget, cairo_t *cr, gpointer data) {
    // White background
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_paint(cr);

    // Draw the grid cells
    for (int row = 0; row < GRID_SIZE; row++) {
        for (int col = 0; col < GRID_SIZE; col++) {
            if (grid[row][col] == 1) {
                cairo_set_source_rgb(cr, 0, 0, 0); // Black
            } else {
                cairo_set_source_rgb(cr, 1, 1, 1); // White
            }
            cairo_rectangle(cr, col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
            cairo_fill(cr);
        }
    }

    // Draw grid lines
    cairo_set_source_rgb(cr, 0.8, 0.8, 0.8); // Light grey for grid lines
    cairo_set_line_width(cr, 0.5);
    for (int i = 1; i < GRID_SIZE; i++) {
        cairo_move_to(cr, i * CELL_SIZE, 0);
        cairo_line_to(cr, i * CELL_SIZE, CANVAS_HEIGHT);
        cairo_move_to(cr, 0, i * CELL_SIZE);
        cairo_line_to(cr, CANVAS_WIDTH, i * CELL_SIZE);
    }
    cairo_stroke(cr);

    return FALSE;
}

/**
 * @brief Helper function to update the grid cell under the cursor.
 */
static void draw_grid_cell(gdouble x, gdouble y) {
    int col = x / CELL_SIZE;
    int row = y / CELL_SIZE;

    if (col >= 0 && col < GRID_SIZE && row >= 0 && row < GRID_SIZE) {
        grid[row][col] = 1; // Set cell to black
        gtk_widget_queue_draw(drawing_area);
    }
}

/**
 * @brief Handle mouse button press events.
 */
static gboolean button_press_event_cb(GtkWidget *widget, GdkEventButton *event, gpointer data) {
    if (event->button == GDK_BUTTON_PRIMARY) {
        draw_grid_cell(event->x, event->y);
    }
    return TRUE;
}

/**
 * @brief Handle mouse motion events.
 */
static gboolean motion_notify_event_cb(GtkWidget *widget, GdkEventMotion *event, gpointer data) {
    if (event->state & GDK_BUTTON1_MASK) {
        draw_grid_cell(event->x, event->y);
    }
    return TRUE;
}


// --- Image Processing and Prediction ---

typedef struct {
    int min_row, max_row, min_col, max_col;
} BoundingBox;

/**
 * @brief Finds the bounding box of the digit and calculates its center of mass,
 *        then translates the digit to center it in the grid.
 * @param network_input The output array to be filled with the centered image data.
 */
static void preprocess_and_center_image(double* network_input) {
    // 1. Find bounding box and center of mass
    BoundingBox bbox = {GRID_SIZE, -1, GRID_SIZE, -1};
    double total_mass = 0;
    double weighted_row_sum = 0;
    double weighted_col_sum = 0;

    for (int r = 0; r < GRID_SIZE; r++) {
        for (int c = 0; c < GRID_SIZE; c++) {
            if (grid[r][c] == 1) {
                if (r < bbox.min_row) bbox.min_row = r;
                if (r > bbox.max_row) bbox.max_row = r;
                if (c < bbox.min_col) bbox.min_col = c;
                if (c > bbox.max_col) bbox.max_col = c;

                total_mass += 1.0;
                weighted_row_sum += r;
                weighted_col_sum += c;
            }
        }
    }

    // Handle empty canvas
    if (total_mass == 0) {
        for (int i = 0; i < NETWORK_INPUT_SIZE; i++) {
            network_input[i] = 0.0;
        }
        return;
    }

    double com_row = weighted_row_sum / total_mass;
    double com_col = weighted_col_sum / total_mass;

    // 2. Calculate translation offset
    int offset_row = (int)round((GRID_SIZE / 2.0) - com_row);
    int offset_col = (int)round((GRID_SIZE / 2.0) - com_col);

    // 3. Create a new centered grid
    double centered_grid[GRID_SIZE][GRID_SIZE] = {0};
    for (int r = bbox.min_row; r <= bbox.max_row; r++) {
        for (int c = bbox.min_col; c <= bbox.max_col; c++) {
            if (grid[r][c] == 1) {
                int new_row = r + offset_row;
                int new_col = c + offset_col;

                if (new_row >= 0 && new_row < GRID_SIZE && new_col >= 0 && new_col < GRID_SIZE) {
                    centered_grid[new_row][new_col] = 1.0;
                }
            }
        }
    }

    // 4. Flatten the centered grid into the network_input array
    for (int r = 0; r < GRID_SIZE; r++) {
        for (int c = 0; c < GRID_SIZE; c++) {
            network_input[r * GRID_SIZE + c] = centered_grid[r][c];
        }
    }
}


/**
 * @brief Saves a grid represented by a double array to a PGM file.
 */
static void save_grid_as_pgm(const char* filename, double* data) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Could not open %s for writing.\n", filename);
        return;
    }

    fprintf(fp, "P2\n%d %d\n255\n", GRID_SIZE, GRID_SIZE);

    for (int i = 0; i < NETWORK_INPUT_SIZE; i++) {
        fprintf(fp, "%d ", (int)(data[i] * 255));
        if ((i + 1) % GRID_SIZE == 0) {
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
    printf("Saved grid to %s\n", filename);
}


/**
 * @brief Processes the grid data and runs prediction.
 */
static void process_and_predict() {
    double network_input[NETWORK_INPUT_SIZE];

    // 1. Save the raw drawing before processing
    double raw_input[NETWORK_INPUT_SIZE];
    for (int i = 0; i < GRID_SIZE; i++) {
        for (int j = 0; j < GRID_SIZE; j++) {
            raw_input[i * GRID_SIZE + j] = (double)grid[i][j];
        }
    }
    save_grid_as_pgm("drawn_digit_raw.pgm", raw_input);

    // 2. Center the image to match MNIST's format
    preprocess_and_center_image(network_input);
    save_grid_as_pgm("drawn_digit_centered.pgm", network_input);


    // 3. Make a prediction
    if (!net) {
        fprintf(stderr, "Error: Network not loaded.\n");
        gtk_label_set_text(GTK_LABEL(prediction_label), "Error: Network not loaded");
        return;
    }

    int prediction = gann_predict(net, network_input);
    GannError err = gann_get_last_error();
    if (err != GANN_SUCCESS) {
        fprintf(stderr, "Error during prediction: %s\n", gann_error_to_string(err));
        gtk_label_set_text(GTK_LABEL(prediction_label), "Prediction Error");
    } else {
        char prediction_str[50];
        snprintf(prediction_str, sizeof(prediction_str), "Prediction: %d", prediction);
        gtk_label_set_text(GTK_LABEL(prediction_label), prediction_str);
    }
}


// --- Main Application Setup ---

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);

    // --- Create Widgets ---
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Digit Recognizer");
    gtk_window_set_resizable(GTK_WINDOW(window), FALSE);
    gtk_window_set_default_size(GTK_WINDOW(window), CANVAS_WIDTH + 150, CANVAS_HEIGHT);

    drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(drawing_area, CANVAS_WIDTH, CANVAS_HEIGHT);

    GtkWidget *predict_button = gtk_button_new_with_label("Predict");
    GtkWidget *clear_button = gtk_button_new_with_label("Clear");
    GtkWidget *load_model_button = gtk_button_new_with_label("Load Model");
    prediction_label = gtk_label_new("Prediction: -");
    model_status_label = gtk_label_new("Model: -"); // Initial text

    // Attempt to load the default neural network at the start
    const char* data_prefix = find_data_path_prefix();
    char network_path[256];
    snprintf(network_path, sizeof(network_path), "%s%s", data_prefix, NETWORK_FILE);
    load_network(network_path);
    if (!net) {
        fprintf(stderr, "INFO: Could not load the default network from '%s'.\n", network_path);
        fprintf(stderr, "You can load a network using the 'Load Model' button.\n");
    }

    // --- Layout ---
    GtkWidget *main_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);
    GtkWidget *controls_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);

    gtk_box_pack_start(GTK_BOX(main_hbox), drawing_area, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(main_hbox), controls_vbox, FALSE, TRUE, 0);

    // Pack buttons into the controls box
    gtk_box_pack_start(GTK_BOX(controls_vbox), predict_button, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(controls_vbox), clear_button, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(controls_vbox), load_model_button, FALSE, FALSE, 5);

    // Add a separator
    GtkWidget *separator = gtk_separator_new(GTK_ORIENTATION_HORIZONTAL);
    gtk_box_pack_start(GTK_BOX(controls_vbox), separator, FALSE, TRUE, 10);

    // Pack labels into the controls box
    gtk_box_pack_start(GTK_BOX(controls_vbox), prediction_label, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(controls_vbox), model_status_label, FALSE, FALSE, 5);

    gtk_container_add(GTK_CONTAINER(window), main_hbox);

    // --- Connect Signals ---
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    g_signal_connect(drawing_area, "draw", G_CALLBACK(draw_cb), NULL);
    g_signal_connect(drawing_area, "motion-notify-event", G_CALLBACK(motion_notify_event_cb), NULL);
    g_signal_connect(drawing_area, "button-press-event", G_CALLBACK(button_press_event_cb), NULL);
    g_signal_connect(predict_button, "clicked", G_CALLBACK(predict_button_clicked), NULL);
    g_signal_connect(clear_button, "clicked", G_CALLBACK(clear_button_clicked), NULL);
    g_signal_connect(load_model_button, "clicked", G_CALLBACK(load_model_button_clicked), window);

    // Enable mouse events on the drawing area
    gtk_widget_set_events(drawing_area, gtk_widget_get_events(drawing_area) | GDK_BUTTON_PRESS_MASK | GDK_POINTER_MOTION_MASK);

    // --- Show and Run ---
    gtk_widget_show_all(window);
    gtk_main();

    // --- Cleanup ---
    if (net) {
        nn_free(net);
    }

    return 0;
}
