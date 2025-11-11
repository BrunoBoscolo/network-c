#include "gann_docs.h"
#include "gann_errors.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "parson.h"

char* gann_get_doc(const char* function_name, const char* lang) {
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "data/%s.json", lang);

    JSON_Value *root_value = json_parse_file(filepath);
    if (json_value_get_type(root_value) != JSONObject) {
        return NULL;
    }

    JSON_Object *root_object = json_value_get_object(root_value);
    JSON_Array *functions = json_object_get_array(root_object, "functions");

    for (size_t i = 0; i < json_array_get_count(functions); i++) {
        JSON_Object *function_obj = json_array_get_object(functions, i);
        const char *name = json_object_get_string(function_obj, "name");
        if (strcmp(name, function_name) == 0) {
            const char *doc = json_object_get_string(function_obj, "doc");
            char* doc_copy = (char*)malloc(strlen(doc) + 1);
            if (doc_copy) {
                strcpy(doc_copy, doc);
            }
            json_value_free(root_value);
            return doc_copy;
        }
    }

    json_value_free(root_value);
    gann_set_error(GANN_ERROR_DOCS_NOT_FOUND);
    return NULL;
}
