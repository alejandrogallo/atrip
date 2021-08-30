(require 'subr-x)

(defun f-join (&rest args)
  (string-join args "/"))

(defvar atrip-sources nil)
(defun atrip-print-sources ()
  (princ (string-join atrip-sources " ")))

(defmacro atrip-def (name body) `(progn (defun ,name () ,body)
                                             (push (,name) atrip-sources)))

;; atrip variables for the org-mode file
(atrip-def atrip-include-f "include/atrip")
(atrip-def atrip-slice-h (f-join (atrip-include-f) "Slice.hpp"))
(atrip-def atrip-utils-h (f-join (atrip-include-f) "Utils.hpp"))
(atrip-def atrip-main-h   "include/atrip.hpp")
