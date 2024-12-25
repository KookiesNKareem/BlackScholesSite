document.addEventListener('DOMContentLoaded', () => {
  const tickerInput = document.getElementById('ticker');
  const currentPriceInput = document.getElementById('current_price');
  const expirationSelect = document.getElementById('expiration_date');
  const optionTypeSelect = document.getElementById('option_type');
  const strikePriceSelect = document.getElementById('strike_price');
  const riskFreeRateInput = document.getElementById('risk_free_rate');

  const form = document.getElementById('option-form');
  const resultColumn = document.getElementById('result-column');
  const resultContainer = document.getElementById('result-container');
  const mainContainer = document.getElementById('main-container');

  let globalExpiriesData = {};

  async function fetchOptionData(symbol) {
    try {
      const response = await fetch(`/api/option_data?symbol=${symbol}`);
      if (!response.ok) {
        throw new Error(`Error: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Fetch error:", error);
      return null;
    }
  }

  function populateStrikes(expiry) {
    strikePriceSelect.innerHTML = '';
    const strikes = globalExpiriesData[expiry] || [];
    if (!strikes.length) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No Strikes';
      strikePriceSelect.appendChild(opt);
      return;
    }
    strikes.forEach((s) => {
      const opt = document.createElement('option');
      opt.value = s;
      opt.textContent = s;
      strikePriceSelect.appendChild(opt);
    });
  }

  tickerInput.addEventListener('blur', async () => {
    const symbol = tickerInput.value.trim();
    if (!symbol) return;

    const data = await fetchOptionData(symbol);
    if (!data || data.error) {
      console.error("Could not fetch option data:", data ? data.error : 'Unknown error');
      return;
    }

    if (typeof data.price === 'number') {
      currentPriceInput.value = data.price.toFixed(2);
    }

    if (typeof data.risk_free_rate === 'number') {
      riskFreeRateInput.value = data.risk_free_rate.toFixed(2);
    }

    globalExpiriesData = data.expiriesData || {};
    expirationSelect.innerHTML = '';
    const expiries = Object.keys(globalExpiriesData);

    if (!expiries.length) {
      const noExp = document.createElement('option');
      noExp.value = '';
      noExp.textContent = 'No Expiries';
      expirationSelect.appendChild(noExp);
      strikePriceSelect.innerHTML = '';
      return;
    }

    expiries.forEach(exp => {
      const opt = document.createElement('option');
      opt.value = exp;
      opt.textContent = exp;
      expirationSelect.appendChild(opt);
    });

    populateStrikes(expiries[0]);
  });

  expirationSelect.addEventListener('change', () => {
    const chosenExpiry = expirationSelect.value;
    populateStrikes(chosenExpiry);
  });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    const formData = new FormData();
    formData.append('ticker', tickerInput.value.trim());
    formData.append('expiration_date', expirationSelect.value);
    formData.append('option_type', optionTypeSelect.value);
    formData.append('strike_price', strikePriceSelect.value);
    formData.append('risk_free_rate', riskFreeRateInput.value);

    try {
      const response = await fetch('/calculate', {
        method: 'POST',
        body: formData
      });
      if (!response.ok) {
        throw new Error(`Server error: ${response.statusText}`);
      }

      const resultHtml = await response.text();
      resultContainer.innerHTML = resultHtml;

      resultColumn.classList.remove('hidden');

      mainContainer.classList.add('slide-left');

    } catch (err) {
      console.error("Error in /calculate:", err);
      resultContainer.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
      resultColumn.classList.remove('hidden');
      mainContainer.classList.add('slide-left');
    }
  });
});