document.addEventListener('DOMContentLoaded', function () {
    // Form Validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });

    // AJAX Requests for API interactions
    const reviewForm = document.getElementById('review-form');
    if (reviewForm) {
        reviewForm.addEventListener('submit', function (event) {
            event.preventDefault();
            const formData = new FormData(reviewForm);
            const loadingIndicator = document.getElementById('loading-indicator');
            if (loadingIndicator) loadingIndicator.style.display = 'block';

            fetch('/api/add_review', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Review added successfully!');
                    location.reload();
                } else {
                    alert('Error adding review: ' + data.error);
                }
            })
            .catch(error => console.error('Error:', error))
            .finally(() => {
                if (loadingIndicator) loadingIndicator.style.display = 'none';
            });
        });
    }

    // Update Review
    const updateReviewForms = document.querySelectorAll('.update-review-form');
    updateReviewForms.forEach(form => {
        form.addEventListener('submit', function (event) {
            event.preventDefault();
            const formData = new FormData(form);
            const loadingIndicator = document.getElementById('loading-indicator');
            if (loadingIndicator) loadingIndicator.style.display = 'block';

            fetch('/api/update_review', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Review updated successfully!');
                    location.reload();
                } else {
                    alert('Error updating review: ' + data.error);
                }
            })
            .catch(error => console.error('Error:', error))
            .finally(() => {
                if (loadingIndicator) loadingIndicator.style.display = 'none';
            });
        });
    });

    // Delete Review
    const deleteReviewButtons = document.querySelectorAll('.delete-review-button');
    deleteReviewButtons.forEach(button => {
        button.addEventListener('click', function () {
            const reviewId = this.dataset.reviewId;
            const loadingIndicator = document.getElementById('loading-indicator');
            if (loadingIndicator) loadingIndicator.style.display = 'block';

            fetch('/api/delete_review', {
                method: 'POST',
                body: new URLSearchParams({ review_id: reviewId })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    alert('Review deleted successfully!');
                    location.reload();
                } else {
                    alert('Error deleting review: ' + data.error);
                }
            })
            .catch(error => console.error('Error:', error))
            .finally(() => {
                if (loadingIndicator) loadingIndicator.style.display = 'none';
            });
        });
    });

    // Dynamic updates and client-side sorting
    const orderBySelect = document.getElementById('order_by');
    if (orderBySelect) {
        orderBySelect.addEventListener('change', function () {
            const loadingIndicator = document.getElementById('loading-indicator');
            if (loadingIndicator) loadingIndicator.style.display = 'block';

            fetch(`/restaurants?order_by=${encodeURIComponent(orderBySelect.value)}`)
            .then(response => response.text())
            .then(html => {
                document.getElementById('restaurant-table-container').innerHTML = html;
            })
            .catch(error => console.error('Error:', error))
            .finally(() => {
                if (loadingIndicator) loadingIndicator.style.display = 'none';
            });
        });
    }

    // Star Rating System
    const starRatingElements = document.querySelectorAll('.star-rating');
    starRatingElements.forEach(ratingElement => {
        const stars = ratingElement.querySelectorAll('.star');
        stars.forEach((star, index) => {
            star.addEventListener('mouseover', () => {
                stars.forEach((s, i) => {
                    s.src = i <= index ? '/static/imgs/star-on.png' : '/static/imgs/star-off.png';
                });
            });

            star.addEventListener('mouseout', () => {
                stars.forEach((s, i) => {
                    s.src = s.classList.contains('selected') ? '/static/imgs/star-on.png' : '/static/imgs/star-off.png';
                });
            });

            star.addEventListener('click', (event) => {
                stars.forEach((s, i) => {
                    s.classList.toggle('selected', i <= index);
                });
                ratingElement.querySelector('input').value = index + 1; // Update hidden input
                event.stopPropagation(); // Prevent popup from closing
                // Show popup if not already displayed
                document.getElementById('popup').classList.add('show');
                document.body.style.overflow = 'hidden';
            });
        });
    });

    // Popup handling
    const popups = document.querySelectorAll('.popup');
    const openPopupButtons = document.querySelectorAll('.open-popup');
    const closeButtons = document.querySelectorAll('.popup .close-button');

    openPopupButtons.forEach(button => {
        button.addEventListener('click', () => {
            const popupId = button.dataset.popupId;
            document.getElementById(popupId).classList.add('show');
            document.body.style.overflow = 'hidden';
        });
    });

    closeButtons.forEach(button => {
        button.addEventListener('click', () => {
            button.closest('.popup').classList.remove('show');
            document.body.style.overflow = '';
        });
    });

    // Ensure popup content is responsive
    popups.forEach(popup => {
        const popupContent = popup.querySelector('.popup-content');
        popupContent.style.maxHeight = window.innerHeight * 0.8 + 'px';
    });

    // Confirm Rating Button
    const confirmRatingButton = document.getElementById('confirm-rating-btn');
    if (confirmRatingButton) {
        confirmRatingButton.addEventListener('click', () => {
            const rating = document.querySelector('#rating').value;
            if (rating > 0) {
                alert('Rating confirmed: ' + rating);
                document.getElementById('popup').classList.remove('show');
                document.body.style.overflow = '';
            }
        });
    }
});
