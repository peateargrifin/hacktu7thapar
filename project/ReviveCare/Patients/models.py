# patient/models.py
from django.db import models
from django.utils import timezone

class Patient(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)   # 🔑 UNIQUE - This is the only authentication
    info = models.TextField(max_length=3000)
    created_at = models.DateTimeField(default=timezone.now)  # Changed from auto_now_add
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name
    
    class Meta:
        ordering = ['-updated_at']
        verbose_name = 'Patient'
        verbose_name_plural = 'Patients'
