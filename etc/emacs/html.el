(require 'package)

(setq package-enable-at-startup t)
(setq package-archives
  '(("gnu"   . "http://elpa.gnu.org/packages/")
    ("melpa" . "http://melpa.org/packages/"   )
    ("org"   . "http://orgmode.org/elpa/"     )))
(package-initialize)

(unless (package-installed-p 'use-package)
  (package-refresh-contents)
  (package-install 'use-package))

(eval-when-compile
  (require 'use-package))

(use-package htmlize
             :ensure t)

(require 'org)

;(use-package org-plus-contrib
;             :ensure 

;(use-package org-src
             ;:ensure t)

(setq org-src-fontify-natively t
      org-src-preserve-indentation t
      org-src-tab-acts-natively t)

(load-theme 'tsdh-light)
(require 'ox-html)
