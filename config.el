(require 'subr-x)
(require 'org)

(defun f-join (&rest args)
  (string-join args "/"))

(defvar atrip-sources nil)
(defun atrip-print-sources ()
  (princ (string-join atrip-sources " ")))

(defvar atrip-include-f "include/atrip") ;; TODO: create defvar
(defvar atrip-src-f "src/atrip")         ;; TODO: create defvar

(defmacro atrip-def (name body) `(progn (defun ,name () ,body)
                                             (push (,name) atrip-sources)))


(defmacro atrip-def-src (name body)
  `(atrip-def ,name (f-join atrip-src-f ,body)))
(defmacro atrip-def-hdr (name body)
  `(atrip-def ,name (f-join atrip-include-f ,body)))

;; atrip variables for the org-mode file
(atrip-def-hdr atrip-slice-h "Slice.hpp")
(atrip-def-hdr atrip-slice-union-h "SliceUnion.hpp")
(atrip-def-hdr atrip-utils-h "Utils.hpp")
(atrip-def-hdr atrip-blas-h "Blas.hpp")
(atrip-def-hdr atrip-rankmap-h "RankMap.hpp")
(atrip-def-hdr atrip-unions-h "Unions.hpp")
(atrip-def-hdr atrip-tuples-h "Tuples.hpp")
(atrip-def-hdr atrip-equations-h "Equations.hpp")
(atrip-def-hdr atrip-debug-h "Debug.hpp")

(atrip-def-hdr atrip-atrip-h "Atrip.hpp")
(atrip-def-src atrip-atrip-cxx "Atrip.cxx")

(atrip-def atrip-main-h   "include/atrip.hpp")

(defvar atrip-root-directory (file-name-directory load-file-name))
(defvar license-path (format "%s/LICENSE-HEADER" atrip-root-directory))

(add-hook 'org-babel-post-tangle-hook
          (lambda ()
            (goto-char (point-min))
            (insert-file-contents license-path)
            (write-region nil nil (buffer-file-name (current-buffer)))
            (message "license header in: %s" (current-buffer))))
