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

// --- Function Prototypes ---
static void load_network(const char* filename);
static void load_model_button_clicked(GtkWidget *widget, gpointer data);
static gboolean draw_network_cb(GtkWidget *widget, cairo_t *cr, gpointer data);

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

static gboolean draw_network_cb(GtkWidget *widget, cairo_t *cr, gpointer data) {
    // White background
    cairo_set_source_rgb(cr, 1, 1, 1);
    cairo_paint(cr);

    if (!net) {
        return FALSE;
    }

    // --- Find min/max weights and biases for scaling ---
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
    int width = gtk_widget_get_allocated_width(widget);
    int height = gtk_widget_get_allocated_height(widget);
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
        }
    }

    return FALSE;
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
    model_status_label = gtk_label_new("Model: -"); // Initial text

    // --- Layout ---
    GtkWidget *main_vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 5);
    GtkWidget *controls_hbox = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 5);

    gtk_box_pack_start(GTK_BOX(main_vbox), controls_hbox, FALSE, TRUE, 0);
    gtk_box_pack_start(GTK_BOX(main_vbox), drawing_area, TRUE, TRUE, 0);

    // Pack buttons into the controls box
    gtk_box_pack_start(GTK_BOX(controls_hbox), load_model_button, FALSE, FALSE, 5);
    gtk_box_pack_start(GTK_BOX(controls_hbox), model_status_label, FALSE, FALSE, 5);

    gtk_container_add(GTK_CONTAINER(window), main_vbox);

    // --- Connect Signals ---
    g_signal_connect(window, "destroy", G_CALLBACK(gtk_main_quit), NULL);
    g_signal_connect(drawing_area, "draw", G_CALLBACK(draw_network_cb), NULL);
    g_signal_connect(load_model_button, "clicked", G_CALLBACK(load_model_button_clicked), window);


    // --- Show and Run ---
    gtk_widget_show_all(window);
    gtk_main();

    // --- Cleanup ---
    if (net) {
        nn_free(net);
    }

    return 0;
}
