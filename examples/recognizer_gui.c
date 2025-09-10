#include <gtk/gtk.h>
#include <gdk/gdk.h>
#include "gann.h"
#include <stdio.h>

// --- Constants ---
#define CANVAS_WIDTH 280
#define CANVAS_HEIGHT 280
#define DOWNSCALE_WIDTH 28
#define DOWNSCALE_HEIGHT 28
#define NETWORK_INPUT_SIZE (DOWNSCALE_WIDTH * DOWNSCALE_HEIGHT)
const char* NETWORK_FILE = "trained_network.dat";

// --- Global Variables ---
static cairo_surface_t *surface = NULL;
static GtkWidget *drawing_area;
static GtkWidget *prediction_label;
static GtkWidget *model_status_label;
static NeuralNetwork* net = NULL;

// --- Function Prototypes ---
static void clear_surface();
static void process_and_predict();
static void load_network(const char* filename);
static void load_model_button_clicked(GtkWidget *widget, gpointer data);

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
 * @brief Clears the drawing surface to white.
 */
static void clear_surface() {
    cairo_t *cr = cairo_create(surface);
    cairo_set_source_rgb(cr, 1, 1, 1); // White
    cairo_paint(cr);
    cairo_destroy(cr);
    gtk_widget_queue_draw(drawing_area); // Redraw the drawing area
}

/**
 * @brief Callback for the "Clear" button.
 */
static void clear_button_clicked(GtkWidget *widget, gpointer data) {
    clear_surface();
    gtk_label_set_text(GTK_LABEL(prediction_label), "Prediction: -");
}

/**
 * @brief Callback for the "Predict" button.
 */
static void predict_button_clicked(GtkWidget *widget, gpointer data) {
    process_and_predict();
}

/**
 * @brief Create a new surface of the appropriate size and clear it to white.
 */
static gboolean configure_event_cb(GtkWidget *widget, GdkEventConfigure *event, gpointer data) {
    if (surface) {
        cairo_surface_destroy(surface);
    }
    surface = gdk_window_create_similar_surface(gtk_widget_get_window(widget),
                                                CAIRO_CONTENT_COLOR,
                                                gtk_widget_get_allocated_width(widget),
                                                gtk_widget_get_allocated_height(widget));
    clear_surface();
    return TRUE;
}

/**
 * @brief Redraw the screen from the surface.
 */
static gboolean draw_cb(GtkWidget *widget, cairo_t *cr, gpointer data) {
    cairo_set_source_surface(cr, surface, 0, 0);
    cairo_paint(cr);
    return FALSE;
}

/**
 * @brief Helper function to draw a brush stroke.
 */
static void draw_brush(GtkWidget *widget, gdouble x, gdouble y) {
    cairo_t *cr = cairo_create(surface);
    cairo_set_source_rgb(cr, 0, 0, 0); // Black
    cairo_rectangle(cr, x - 10, y - 10, 20, 20); // Draw a 20x20 square brush
    cairo_fill(cr);
    cairo_destroy(cr);
    gtk_widget_queue_draw_area(widget, x - 10, y - 10, 20, 20); // Update the affected area
}

/**
 * @brief Handle mouse button press events.
 */
static gboolean button_press_event_cb(GtkWidget *widget, GdkEventButton *event, gpointer data) {
    if (event->button == GDK_BUTTON_PRIMARY) {
        draw_brush(widget, event->x, event->y);
    }
    return TRUE;
}

/**
 * @brief Handle mouse motion events.
 */
static gboolean motion_notify_event_cb(GtkWidget *widget, GdkEventMotion *event, gpointer data) {
    if (event->state & GDK_BUTTON1_MASK) {
        draw_brush(widget, event->x, event->y);
    }
    return TRUE;
}


// --- Image Processing and Prediction ---

/**
 * @brief Processes the drawing on the canvas and runs prediction.
 */
static void process_and_predict() {
    // 1. Get the pixel data from the Cairo surface
    GdkPixbuf *pixbuf = gdk_pixbuf_get_from_surface(surface, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    if (!pixbuf) {
        fprintf(stderr, "Error: Failed to get pixbuf from surface.\n");
        return;
    }

    // 2. Downscale the image to 28x28
    GdkPixbuf *scaled_pixbuf = gdk_pixbuf_scale_simple(pixbuf, DOWNSCALE_WIDTH, DOWNSCALE_HEIGHT, GDK_INTERP_BILINEAR);
    g_object_unref(pixbuf); // Free original pixbuf
    if (!scaled_pixbuf) {
        fprintf(stderr, "Error: Failed to scale pixbuf.\n");
        return;
    }

    // 3. Convert to grayscale and normalize into a flat array
    double network_input[NETWORK_INPUT_SIZE];
    guchar *pixels = gdk_pixbuf_get_pixels(scaled_pixbuf);
    int n_channels = gdk_pixbuf_get_n_channels(scaled_pixbuf);
    int rowstride = gdk_pixbuf_get_rowstride(scaled_pixbuf);

    for (int y = 0; y < DOWNSCALE_HEIGHT; y++) {
        for (int x = 0; x < DOWNSCALE_WIDTH; x++) {
            guchar *p = pixels + y * rowstride + x * n_channels;
            // Simple grayscale conversion: average R, G, B
            double grayscale = (p[0] + p[1] + p[2]) / 3.0;
            // Normalize and invert: network expects white digit on black background
            network_input[y * DOWNSCALE_WIDTH + x] = (255.0 - grayscale) / 255.0;
        }
    }

    g_object_unref(scaled_pixbuf); // Free scaled pixbuf

    // 4. Make a prediction
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
    gtk_window_set_default_size(GTK_WINDOW(window), CANVAS_WIDTH, CANVAS_HEIGHT + 50);

    drawing_area = gtk_drawing_area_new();
    gtk_widget_set_size_request(drawing_area, CANVAS_WIDTH, CANVAS_HEIGHT);

    GtkWidget *predict_button = gtk_button_new_with_label("Predict");
    GtkWidget *clear_button = gtk_button_new_with_label("Clear");
    GtkWidget *load_model_button = gtk_button_new_with_label("Load Model");
    prediction_label = gtk_label_new("Prediction: -");
    model_status_label = gtk_label_new("Model: -"); // Initial text

    // Attempt to load the default neural network at the start
    load_network(NETWORK_FILE);
    if (!net) {
        fprintf(stderr, "INFO: Could not load the default network from '%s'.\n", NETWORK_FILE);
        fprintf(stderr, "You can load a network using the 'Load Model' button.\n");
    }

    // --- Layout ---
    GtkWidget *vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    GtkWidget *hbox_buttons = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);
    GtkWidget *hbox_status = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);

    gtk_box_pack_start(GTK_BOX(vbox), drawing_area, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(vbox), hbox_buttons, FALSE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(vbox), hbox_status, FALSE, TRUE, 0);

    // Pack buttons
    gtk_box_pack_start(GTK_BOX(hbox_buttons), predict_button, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(hbox_buttons), clear_button, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(hbox_buttons), load_model_button, TRUE, TRUE, 0);

    // Pack labels
    gtk_box_pack_start(GTK_BOX(hbox_status), prediction_label, TRUE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(hbox_status), model_status_label, TRUE, TRUE, 0);


    gtk_container_add(GTK_CONTAINER(window), vbox);

    // --- Connect Signals ---
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    g_signal_connect(drawing_area, "draw", G_CALLBACK(draw_cb), NULL);
    g_signal_connect(drawing_area, "configure-event", G_CALLBACK(configure_event_cb), NULL);
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
    if (surface) {
        cairo_surface_destroy(surface);
    }
    if (net) {
        nn_free(net);
    }

    return 0;
}
