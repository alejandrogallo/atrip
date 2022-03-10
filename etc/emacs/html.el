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

(use-package ox-rst
             :ensure t)

(require 'org)

(setq org-src-fontify-natively t
      org-src-preserve-indentation t
      org-src-tab-acts-natively t)

(load-theme 'tsdh-light)
(require 'ox-html)
(require 'ox-rst)
