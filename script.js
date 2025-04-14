document.getElementById('csvFile').addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (!file) return;
  
    const status = document.getElementById('status');
    const table = document.getElementById('resultsTable');
    const tbody = table.querySelector('tbody');
  
    status.textContent = '⏳ Predicting...';
    table.classList.add('hidden');
    tbody.innerHTML = '';
  
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      transform: (value) => {
        if (value === '') return null;
        if (!isNaN(value) && value.trim() !== '') return Number(value);
        if (value.toLowerCase() === 'true') return true;
        if (value.toLowerCase() === 'false') return false;
        return value;
      },
      complete: async (results) => {
        const rows = results.data;
  
        for (const row of rows) {
          let prediction = '-', probas = '-', error = '-';
  
          try {
            const response = await fetch('http://localhost:5000/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(row)
            });
  
            if (!response.ok) throw new Error(await response.text());
  
            const data = await response.json();
            prediction = data.prediction;
            probas = data.prediction_proba
              .map((p, i) => `Class ${i}: ${(p * 100).toFixed(1)}%`)
              .join(', ');
          } catch (err) {
            error = err.message || 'Request failed';
          }
  
          const tr = document.createElement('tr');
  
          tr.innerHTML = `
            <td><pre>${JSON.stringify(row, null, 2)}</pre></td>
            <td>${prediction}</td>
            <td>${probas}</td>
            <td class="error">${error}</td>
          `;
  
          tbody.appendChild(tr);
        }
  
        status.textContent = '✅ Done!';
        table.classList.remove('hidden');
      }
    });
  });
  