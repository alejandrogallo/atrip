#+quicklisp
(eval-when (:compile-toplevel :load-toplevel :execute)
  (ql:quickload '(vgplot fiveam)))

(defpackage :naive-tuples
  (:use :cl :vgplot))
(in-package :naive-tuples)

(defun tuples-atrip (nv)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (loop :for a :below nv
        :append
        (loop :for b :from a :below nv
              :append
              (loop :for c :from b :below nv
                    :unless (= a b c)
                      :collect (list a b c)))))

(defun tuples-half (nv)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (loop :for a :below nv
        :append
        (loop :for b :from a :below nv
              :append
              (loop :for c :from b :below nv
                    :collect (list a b c)))))

(defun tuples-all (nv)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (loop :for a :below nv
        :append
        (loop :for b :below nv
              :append
              (loop :for c :below nv
                    :collect (list a b c)))))

(defun tuples-all-nth (i nv)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (list (floor i (* nv nv))
        (mod (floor i nv) nv)
        (mod i nv)))


(defparameter tups (tuples-all 10))

(defun compare-all (l)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((tups (tuples-all l)))
    (loop for i below (length tups)
          do (let* ((good (nth i tups))
                    (bad (tuples-all-nth i l))
                    (eq? (equal good bad)))
               (unless eq?
                 (print (list :|i| i
                              :good good
                              :bad bad)))))))


;; (defun a-half (i nv)
;;   (let ((divisor t)
;;         (j i)
;;         (total-blk 0))
;;     (loop :for a :below nv
;;           :unless (eq divisor 0)
;;             :do (let ((blk (a-block a nv)))
;;                   (multiple-value-bind (d r) (floor j blk)
;;                     (declare (ignore r))
;;                     (when (> d 0)
;;                       (incf total-blk blk))
;;                     (setq j (- j blk)
;;                           divisor d)))
;;           :else
;;             :return (values (- a 1)
;;                             i
;;                             total-blk))))

;; (defun b-half (i a nv a-block-sum)
;;   "we have
;;     \begin{equation}
;;       i = \underbrace{B(a_0) +
;;                       \cdots +
;;                       B(a_{i-1})}_{\texttt{a-block-sum}}
;;          + idx
;;     \end{equation}
;;     and with this we just have to divide.
;;   "
;;   (let ((bj (if (> a-block-sum 0)
;;                 (mod i a-block-sum)
;;                 i))
;;         (total-blk 0))
;;     (loop :for b :from a :below Nv
;;           :with divisor = 1
;;           :unless (eq divisor 0)
;;             :do (let ((blk (+ (- nv a)
;;                               #|because|# 1)))
;;                   (incf total-blk blk)
;;                   (if (> blk 0)
;;                       (multiple-value-bind (d r) (floor bj blk)
;;                         (declare (ignore r))
;;                         (setq bj (- bj blk)
;;                               divisor d))
;;                       (setq divisor 0)))
;;           :else
;;             :return (values (- b 1)
;;                             bj
;;                             total-blk))))

(defun a-block (a nv)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (- (* (- nv 1) (- nv (- a 1)))
     (- (floor (* (- nv 1) nv)
               2)
        (floor (* (- a 1) (- a 2))
               2))))

(defun a-block-sum (|t| nv)
  (macrolet ((ssum (n) `(floor (* ,n (+ ,n 1))
                               2))
             (qsum (n) `(floor (* ,n
                                  (+ ,n 1)
                                  (+ 1 (* 2 ,n)))
                               6)))
    (let ((nv-1 (- nv 1))
          (t+1 (+ |t| 1)))
      (+ (* t+1 nv-1 nv)
         (* nv-1 t+1)
         (- (* nv-1
               (ssum |t|)))
         (- (* t+1
               (ssum nv-1)))
         (floor (- (qsum |t|)
                   (* 3 (ssum |t|)))
                2)
         t+1))))

(defun get-half (i nv &key from block)
  (let ((divisor 1)
        (j i)
        (total-blk 0))
    (loop :for α :from from :below nv
          :unless (eq divisor 0)
            :do (let ((blk (funcall block α nv)))
                  (multiple-value-bind (d r) (floor j blk)
                    (declare (ignore r))
                    (when (> d 0)
                      (incf total-blk blk)
                      (setq j (- j blk)))
                    (setq divisor d)))
          :else
            :return (values (- α 1)
                            j
                            total-blk))))

(defun tuples-half-nth (i nv)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (flet ((bc-block (x %nv)
           (+ 1 (- %nv x))))
    (multiple-value-bind (a aj blks) (get-half i nv :from 0 :block #'a-block)
      (declare (ignore blks))
      (multiple-value-bind (b bj blks) (get-half aj nv
                                                 :from a
                                                 :block #'bc-block)
        (declare (ignore blks))
        (multiple-value-bind (c cj blks) (get-half bj nv
                                                   :from b
                                                   :block #'bc-block)
          (declare (ignore cj blks))
          (print (list :idxs aj bj cj))
          (list a b c))))))

(defun a-block-atrip (a nv)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (- (a-block a nv) 1))

(defun a-block-sum-atrip (|t| nv)
  (declare (optimize (speed 3) (safety 0) (debug 0)))
  (- (a-block-sum |t| nv) (+ |t| 1)))

(defun b-block-sum-atrip (a |t| nv)
  (- (* nv
        (1+ (- |t| a)))
     (floor (- (* |t| (1+ |t|))
               (* a (- a 1)))
            2)
     1))

(defun nth-atrip (i nv)
  (let ((sums (mapcar (lambda (s) (a-block-sum-atrip s nv))
                      (loop :for j :below nv :collect j))))
    (multiple-value-bind (a ablk)
        (loop :for sum :in sums
              :with a = -1
              :with base = 0
              :do (incf a)
              :if (eq (floor i sum) 0)
                :return (values a base)
              :else
                :do (setq base sum))
      (multiple-value-bind (b bblk)
          (let ((sums (mapcar (lambda (s)
                                (+ ablk
                                   #+nil(- nv s 1)
                                   (b-block-sum-atrip a s nv)))
                              (loop :for b :from a :below nv
                                    :collect b))))
            (loop :for sum :in sums
                  :with b = (- a 1)
                  :with base = ablk
                  :do (incf b)
                  :if (< i sum)
                    :return (values b base)
                  :else
                    :do (progn
                          ;; (print sums)
                          (setq base sum))))
        (list a b (+ b
                     (- i bblk)
                     (if (eq a b)
                         1
                         0)))))))

(defun atrip-test (i nv)
  (let ((tuples (tuples-atrip nv))
        (cheaper (nth-atrip i nv)))
    (values (nth i tuples)
            cheaper
            (print (equal (nth i tuples)
                   cheaper)))))
