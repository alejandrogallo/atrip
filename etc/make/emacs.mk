EMACS = emacs -q --batch --load config.el
define tangle
$(EMACS) $(1) --eval "(require 'org)" --eval '(org-babel-tangle)'
endef

