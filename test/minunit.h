#ifndef MINUNIT_H
#define MINUNIT_H

#include <stdio.h>
#include <string.h>

#define mu_assert(message, test) do { if (!(test)) return message; } while (0)
#define mu_run_test(test) do { printf("--- Running test: %s\n", #test); const char *message = test(); tests_run++; if (message) return message; } while (0)
#define mu_check(test) mu_assert("Assertion failed", test)
#define mu_assert_string_eq(expected, actual) mu_assert("Strings are not equal", strcmp(expected, actual) == 0)


extern int tests_run;

#endif // MINUNIT_H
