#include "gann_docs.h"
#include "gann_errors.h"
#include "parson.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* gann_get_doc(const char* function_name, const char* lang) {
    if (!function_name || !lang) {
        gann_set_error(GANN_ERROR_NULL_ARGUMENT);
        return NULL;
    }

    char filepath[256];
    snprintf(filepath, sizeof(filepath), "data/%s.json", lang);

    JSON_Value *root_value = json_parse_file(filepath);
    if (json_value_get_type(root_value) != JSONObject) {
        gann_set_error(GANN_ERROR_FILE_READ);
        json_value_free(root_value);
        return NULL;
    }

    JSON_Object *root_object = json_value_get_object(root_value);
    JSON_Array *functions = json_object_get_array(root_object, "functions");
    if (!functions) {
        gann_set_error(GANN_ERROR_INVALID_FILE_FORMAT);
        json_value_free(root_value);
        return NULL;
    }

    for (size_t i = 0; i < json_array_get_count(functions); i++) {
        JSON_Object *function_obj = json_array_get_object(functions, i);
        const char *name = json_object_get_string(function_obj, "name");
        if (name && strcmp(name, function_name) == 0) {
            const char *doc_string = json_object_get_string(function_obj, "doc");
            if (!doc_string) {
                gann_set_error(GANN_ERROR_DOC_NOT_FOUND);
                json_value_free(root_value);
                return NULL;
            }

            char* result = (char*)malloc(strlen(doc_string) + 1);
            if (!result) {
                gann_set_error(GANN_ERROR_ALLOC_FAILED);
                json_value_free(root_value);
                return NULL;
            }

            strcpy(result, doc_string);
            json_value_free(root_value);
            gann_set_error(GANN_SUCCESS);
            return result;
        }
    }

    gann_set_error(GANN_ERROR_DOC_NOT_FOUND);
    json_value_free(root_value);
    return NULL;
}
