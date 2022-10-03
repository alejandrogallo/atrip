;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((c++-mode . ((outline-regexp . "// \\[\\[file:")
              (eval . (let
                          ((root
                            (expand-file-name
                             (project-root
                              (project-current)))))
                        (setq-local flycheck-gcc-include-path
                                    (list
                                     (format "%s/vendor/include/" root)
                                     (format "%s/include/" root)
                                     (format "%s/" root)
                                     (format "%s/bench/" root)
                                     (format "%s/build/main/" root)))))
              (eval . (flycheck-mode))
              (eval . (outline-minor-mode))
              (indent-tabs-mode . nil)
              (tab-width . 2))))
