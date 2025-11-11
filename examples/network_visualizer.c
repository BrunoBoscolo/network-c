#include <gtk/gtk.h>
#include <gdk/gdk.h>
#include "gann.h"
#include <stdio.h>
#include "utils.h"
#include <math.h>

// --- Global Variables ---
static GtkWidget *drawing_area;
static GtkWidget *model_status_label;
static NeuralNetwork* net = NULL;
static double zoom = 1.0;
static double pan_x = 0.0;
static double pan_y = 0.0;
static double drag_start_x = 0;
static double drag_start_y = 0;
static gboolean dragging = FALSE;
static GdkPixbuf *pixbuf = NULL;
static gboolean render_as_image = FALSE;
#define COMPLEXITY_THRESHOLD 5000
#define RENDER_WIDTH 3840
#define RENDER_HEIGHT 2160


// --- Function Prototypes ---
static void load_network(const char* filename);
static void load_model_button_clicked(GtkWidget *widget, gpointer data);
static void force_image_render_button_clicked(GtkWidget *widget, gpointer data);
static gboolean draw_network_cb(GtkWidget *widget, cairo_t *cr, gpointer data);
static void render_network_to_pixbuf();
static int get_network_complexity();
static void draw_network_vector(cairo_t *cr, int width, int height);
static gboolean scroll_event_cb(GtkWidget *widget, GdkEventScroll *event, gpointer data);
static gboolean button_press_event_cb(GtkWidget *widget, GdkEventButton *event, gpointer data);
static gboolean button_release_event_cb(GtkWidget *widget, GdkEventButton *event, gpointer data);
static gboolean motion_notify_event_cb(GtkWidget *widget, GdkEventMotion *event, gpointer data);

// --- GUI Callbacks ---

static void load_network(const char* filename) {
    if (net) {
        nn_free(net);
        net = NULL;
    }
    if (pixbuf) {
        g_object_unref(pixbuf);
        pixbuf = NULL;
    }
    render_as_image = FALSE;

    net = nn_load(filename);

    if (net) {
        char status_text[1024];
        g_snprintf(status_text, sizeof(status_text), "Model: %s", g_path_get_basename(filename));
        gtk_label_set_text(GTK_LABEL(model_status_label), status_text);

        if (get_network_complexity() > COMPLEXITY_THRESHOLD) {
            render_as_image = TRUE;
        }

        gtk_widget_queue_draw(drawing_area); // Trigger a redraw
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

static void force_image_render_button_clicked(GtkWidget *widget, gpointer data) {
    render_as_image = TRUE;
    if (pixbuf) {
        g_object_unref(pixbuf);
        pixbuf = NULL;
    }
    gtk_widget_queue_draw(drawing_area);
}

static int get_network_complexity() {
    if (!net) return 0;

    int complexity = 0;
    for (int i = 0; i < net->num_layers - 1; i++) {
        complexity += net->architecture[i] * net->architecture[i+1];
    }
    return complexity;
}


static gboolean button_press_event_cb(GtkWidget *widget, GdkEventButton *event, gpointer data) {
    if (event->button == GDK_BUTTON_PRIMARY) {
        dragging = TRUE;
        drag_start_x = event->x;
        drag_start_y = event->y;
    }
    return TRUE;
}

static gboolean button_release_event_cb(GtkWidget *widget, GdkEventButton *event, gpointer data) {
    if (event->button == GDK_BUTTON_PRIMARY) {
        dragging = FALSE;
    }
    return TRUE;
}

static gboolean motion_notify_event_cb(GtkWidget *widget, GdkEventMotion *event, gpointer data) {
    if (dragging) {
        pan_x += (event->x - drag_start_x) / zoom;
        pan_y += (event->y - drag_start_y) / zoom;
        drag_start_x = event->x;
        drag_start_y = event->y;
        gtk_widget_queue_draw(widget);
    }
    return TRUE;
}


static gboolean scroll_event_cb(GtkWidget *widget, GdkEventScroll *event, gpointer data) {
    double old_zoom = zoom;
    if (event->direction == GDK_SCROLL_UP) {
        zoom *= 1.1;
    } else if (event->direction == GDK_SCROLL_DOWN) {
        zoom /= 1.1;
    }

    // Zoom relative to the mouse cursor
    pan_x = event->x / old_zoom - event->x / zoom + pan_x;
    pan_y = event->y / old_zoom - event->y / zoom + pan_y;


    gtk_widget_queue_draw(widget);
    return TRUE;
}

static void render_network_to_pixbuf() {
    if (!net || !drawing_area) return;

    int width = RENDER_WIDTH;
    int height = RENDER_HEIGHT;

    cairo_surface_t *surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, width, height);
    cairo_t *cr = cairo_create(surface);

    draw_network_vector(cr, width, height);

    if (pixbuf) {
        g_object_unref(pixbuf);
    }
    pixbuf = gdk_pixbuf_get_from_surface(surface, 0, 0, width, height);

    cairo_destroy(cr);
    cairo_surface_destroy(surface);
}


static gboolean draw_network_cb(GtkWidget *widget, cairo_t *cr, gpointer data) {
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_paint(cr);

    if (!net) {
        return FALSE;
    }

    if (render_as_image) {
        if (!pixbuf) {
            render_network_to_pixbuf();
        }
        if (pixbuf) {
            cairo_save(cr);
            double pixbuf_width = gdk_pixbuf_get_width(pixbuf);
            double widget_width = gtk_widget_get_allocated_width(widget);
            double scale_factor = widget_width / pixbuf_width;

            cairo_translate(cr, pan_x * zoom, pan_y * zoom);
            cairo_scale(cr, zoom * scale_factor, zoom * scale_factor);
            gdk_cairo_set_source_pixbuf(cr, pixbuf, 0, 0);
            cairo_paint(cr);
            cairo_restore(cr);
        }
    } else {
        cairo_save(cr);
        cairo_translate(cr, pan_x * zoom, pan_y * zoom);
        cairo_scale(cr, zoom, zoom);
        draw_network_vector(cr, gtk_widget_get_allocated_width(widget), gtk_widget_get_allocated_height(widget));
        cairo_restore(cr);
    }

    return FALSE;
}

static void draw_network_vector(cairo_t *cr, int width, int height) {
     cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_paint(cr);
    double max_abs_weight = 0;
    double max_abs_bias = 0;
    for (int i = 0; i < net->num_layers - 1; i++) {
        for (int r = 0; r < net->weights[i]->rows; r++) {
            for (int c = 0; c < net->weights[i]->cols; c++) {
                if (fabs(net->weights[i]->data[r][c]) > max_abs_weight) {
                    max_abs_weight = fabs(net->weights[i]->data[r][c]);
                }
            }
        }
        for (int r = 0; r < net->biases[i]->rows; r++) {
            for (int c = 0; c < net->biases[i]->cols; c++) {
                if (fabs(net->biases[i]->data[r][c]) > max_abs_bias) {
                    max_abs_bias = fabs(net->biases[i]->data[r][c]);
                }
            }
        }
    }


    // --- Drawing Parameters ---
    int padding = 50;
    int layer_spacing = (net->num_layers > 1) ? (width - 2 * padding) / (net->num_layers - 1) : 0;
    double neuron_radius = 10;

    // --- Draw Connections ---
    for (int i = 0; i < net->num_layers - 1; i++) {
        int neurons_in_layer = net->architecture[i];
        int neurons_in_next_layer = net->architecture[i+1];
        double layer_x = padding + i * layer_spacing;
        double next_layer_x = padding + (i + 1) * layer_spacing;

        for (int j = 0; j < neurons_in_next_layer; j++) {
            double next_neuron_y = padding + j * (height - 2 * padding) / (neurons_in_next_layer - 1);
            if (neurons_in_next_layer == 1) next_neuron_y = height / 2;


            for (int k = 0; k < neurons_in_layer; k++) {
                double neuron_y = padding + k * (height - 2 * padding) / (neurons_in_layer - 1);
                 if (neurons_in_layer == 1) neuron_y = height / 2;

                double weight = net->weights[i]->data[k][j];
                double line_width = (max_abs_weight > 0) ? (fabs(weight) / max_abs_weight) * 5.0 : 0.5;

                if (weight > 0) {
                    cairo_set_source_rgba(cr, 0, 0, 1, 0.5); // Blue for positive weights
                } else {
                    cairo_set_source_rgba(cr, 1, 0, 0, 0.5); // Red for negative weights
                }

                cairo_set_line_width(cr, line_width);
                cairo_move_to(cr, layer_x, neuron_y);
                cairo_line_to(cr, next_layer_x, next_neuron_y);
                cairo_stroke(cr);

                if (zoom > 5.0) {
                    char weight_str[16];
                    snprintf(weight_str, sizeof(weight_str), "%.2f", weight);
                    cairo_save(cr);
                    cairo_set_source_rgb(cr, 0, 0, 0);
                    cairo_move_to(cr, (layer_x + next_layer_x) / 2, (neuron_y + next_neuron_y) / 2);
                    cairo_show_text(cr, weight_str);
                    cairo_restore(cr);
                }
            }
        }
    }


    // --- Draw Neurons ---
    for (int i = 0; i < net->num_layers; i++) {
        int neurons_in_layer = net->architecture[i];
        double layer_x = padding + i * layer_spacing;

        for (int j = 0; j < neurons_in_layer; j++) {
            double neuron_y = padding + j * (height - 2 * padding) / (neurons_in_layer - 1);
            if (neurons_in_layer == 1) neuron_y = height / 2;


            // Bias visualization
            if (i > 0) {
                double bias = net->biases[i-1]->data[0][j];
                double bias_strength = (max_abs_bias > 0) ? fabs(bias) / max_abs_bias : 0;
                if (bias > 0) {
                     cairo_set_source_rgb(cr, 1-bias_strength, 1-bias_strength, 1); // Blue for positive bias
                } else {
                     cairo_set_source_rgb(cr, 1, 1-bias_strength, 1-bias_strength); // Red for negative bias
                }

            } else {
                 cairo_set_source_rgb(cr, 1, 1, 1); // White for input layer
            }


            cairo_arc(cr, layer_x, neuron_y, neuron_radius, 0, 2 * M_PI);
            cairo_fill_preserve(cr);
            cairo_set_source_rgb(cr, 0, 0, 0); // Black outline
            cairo_set_line_width(cr, 1.5);
            cairo_stroke(cr);

            if (zoom > 5.0 && i > 0) {
                char bias_str[16];
                snprintf(bias_str, sizeof(bias_str), "%.2f", net->biases[i-1]->data[0][j]);
                cairo_save(cr);
                cairo_set_source_rgb(cr, 0, 0, 0);
                cairo_move_to(cr, layer_x + neuron_radius + 2, neuron_y);
                cairo_show_text(cr, bias_str);
                cairo_restore(cr);
            }
        }
    }
}
// --- Main Application Setup ---

int main(int argc, char *argv[]) {
    gtk_init(&argc, &argv);

    // --- Create Widgets ---
    GtkWidget *window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
    gtk_window_set_title(GTK_WINDOW(window), "Neural Network Visualizer");
    gtk_window_maximize(GTK_WINDOW(window));

    drawing_area = gtk_drawing_area_new();

    GtkWidget *load_model_button = gtk_button_new_with_label("Load Model");
    GtkWidget *force_image_button = gtk_button_new_with_label("Force Image Render");
    model_status_label = gtk_label_new("Model: -"); // Initial text

    // --- Layout ---
    GtkWidget *main_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    GtkWidget *controls_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);

    gtk_box_pack_start(GTK_BOX(main_vbox), controls_hbox, FALSE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(main_vbox), drawing_area, TRUE, TRUE, 0);

    // Pack buttons into the controls box
    gtk_box_pack_start(GTK_BOX(controls_hbox), load_model_button, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(controls_hbox), force_image_button, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(controls_hbox), model_status_label, FALSE, FALSE, 5);

    gtk_container_add(GTK_CONTAINER(window), main_vbox);

    // --- Connect Signals ---
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    g_signal_connect(drawing_area, "draw", G_CALLBACK(draw_network_cb), NULL);
    g_signal_connect(drawing_area, "scroll-event", G_CALLBACK(scroll_event_cb), NULL);
    g_signal_connect(drawing_area, "button-press-event", G_CALLBACK(button_press_event_cb), NULL);
    g_signal_connect(drawing_area, "button-release-event", G_CALLBACK(button_release_event_cb), NULL);
    g_signal_connect(drawing_area, "motion-notify-event", G_CALLBACK(motion_notify_event_cb), NULL);
    g_signal_connect(load_model_button, "clicked", G_CALLBACK(load_model_button_clicked), window);
    g_signal_connect(force_image_button, "clicked", G_CALLBACK(force_image_render_button_clicked), NULL);

    // Enable mouse events
    gtk_widget_set_events(drawing_area, gtk_widget_get_events(drawing_area) | GDK_SCROLL_MASK | GDK_BUTTON_PRESS_MASK | GDK_BUTTON_RELEASE_MASK | GDK_POINTER_MOTION_MASK);


    // --- Show and Run ---
    gtk_widget_show_all(window);
    gtk_main();

    // --- Cleanup ---
    if (net) {
        nn_free(net);
    }
    if (pixbuf) {
        g_object_unref(pixbuf);
    }

    return 0;
}
